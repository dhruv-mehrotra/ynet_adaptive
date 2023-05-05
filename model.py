import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from utils.softargmax import SoftArgmax2D, create_meshgrid
from utils.dataset import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from test import evaluate
from train import train


class StyleModulator(nn.Module):
	def __init__(self, sizes):
		"""
		Additional style modulator for efficient fine-tuning
		"""		
		from ddf import DDFPack
		super(StyleModulator, self).__init__()
		tau = 0.5
		self.modulators = nn.ModuleList(
			[DDFPack(s) for s in sizes + [sizes[-1]]]
		)

	def forward(self, x):
		stylized = []
		for xi, layer in zip(x, self.modulators):
			stylized.append(layer(xi))
		return stylized

class ResNetEncoder(nn.Module):
    def __init__(self, num_layers=18):
        super(ResNetEncoder, self).__init__()
        if num_layers == 18:
            self.resnet = models.resnet18(pretrained=True)
            self.num_channels = [64, 128, 256, 512]
        elif num_layers == 34:
            self.resnet = models.resnet34(pretrained=True)
            self.num_channels = [64, 128, 256, 512]
        elif num_layers == 50:
            self.resnet = models.resnet50(pretrained=True)
            self.num_channels = [256, 512, 1024, 2048]
        elif num_layers == 101:
            self.resnet = models.resnet101(pretrained=True)
            self.num_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError("Invalid number of layers for ResNet")

    def forward(self, x):
        features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        features.append(x)

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        features.append(x)

        x = self.resnet.layer2(x)
        features.append(x)

        x = self.resnet.layer3(x)
        features.append(x)

        x = self.resnet.layer4(x)
        features.append(x)

        return features



class ResNetDecoder(nn.Module):
    def __init__(self, num_channels=[512, 256, 128, 64], output_len=30):
        super(ResNetDecoder, self).__init__()
        self.num_channels = num_channels
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels_),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels_),
                nn.ReLU(inplace=True)
            )
            for in_channels_, out_channels_ in zip(num_channels[:-1], num_channels[1:])
        ])
        self.predictor = nn.Conv2d(num_channels[-1], output_len * 2, kernel_size=1)

    def forward(self, features):
        x = features[-1]
        for i in range(len(self.decoder)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip_connection = features[-i-2]
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder[i](x)
        x = self.predictor(x)
        return x
 

class YNetTorch(nn.Module):
    def __init__(self, num_channels=[64, 128, 256, 512], output_len=30):
        super(YNetTorch, self).__init__()
        self.encoder = ResNetEncoder()
        self.decoder = ResNetDecoder(num_channels=num_channels, output_len=output_len)

    def forward(self, x):
        features = self.encoder(x)
        prediction = self.decoder(features)
        return prediction


class YNet:
	def __init__(self, obs_len, pred_len, params):
		"""
		Ynet class, following a sklearn similar class structure
		:param obs_len: observed timesteps
		:param pred_len: predicted timesteps
		:param params: dictionary with hyperparameters
		"""
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.division_factor = 2 ** len(params['encoder_channels'])

		self.model = YNetTorch(obs_len=obs_len,
							   pred_len=pred_len,
							   segmentation_model_fp=params['segmentation_model_fp'],
							   use_features_only=params['use_features_only'],
							   semantic_classes=params['semantic_classes'],
							   encoder_channels=params['encoder_channels'],
							   decoder_channels=params['decoder_channels'],
							   waypoints=len(params['waypoints']))

	def train(self, train_data, val_data, params, train_image_path, val_image_path, experiment_name, batch_size=8, num_goals=20, num_traj=1, device=None, dataset_name=None, use_raw_data=False, epochs_checkpoints=None, train_net="all", fine_tune=False):
		"""
		Train function
		:param train_data: pd.df, train data
		:param val_data: pd.df, val data
		:param params: dictionary with training hyperparameters
		:param train_image_path: str, filepath to train images
		:param val_image_path: str, filepath to val images
		:param experiment_name: str, arbitrary name to name weights file
		:param batch_size: int, batch size
		:param num_goals: int, number of goals per trajectory, K_e in paper
		:param num_traj: int, number of trajectory per goal, K_a in paper
		:param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
		:return:
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		elif dataset_name == 'ind':
			image_file_name = 'reference.png'
		elif dataset_name == 'eth':
			image_file_name = 'oracle.png'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		# ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
		if dataset_name == 'eth':
			self.homo_mat = {}
			for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
				self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
			seg_mask = True
		else:
			self.homo_mat = None
			seg_mask = False

		# Load train images and augment train data and images
		if fine_tune:
			train_images = create_images_dict(train_data, image_path=train_image_path, image_file=image_file_name, use_raw_data=use_raw_data)
		else:
			train_data, train_images = augment_data(train_data, image_path=train_image_path, image_file=image_file_name,
											  seg_mask=seg_mask, use_raw_data=use_raw_data)

		# Load val scene images
		val_images = create_images_dict(val_data, image_path=val_image_path, image_file=image_file_name, use_raw_data=use_raw_data)

		# Initialize dataloaders
		train_dataset = SceneDataset(train_data, resize=params['resize'], total_len=total_len)
		train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=scene_collate, shuffle=True)

		val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len)
		val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(train_images, factor=params['resize'], seg_mask=seg_mask)
		pad(train_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

		resize(val_images, factor=params['resize'], seg_mask=seg_mask)
		pad(val_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(val_images, seg_mask=seg_mask)

		model = self.model.to(device)

		# Freeze segmentation model
		for param in model.semantic_segmentation.parameters():
			param.requires_grad = False

		if train_net in ["encoder", "modulator"]:
			for param in model.parameters():
				param.requires_grad = False
			if train_net == "encoder":
				for param in model.encoder.parameters():
					param.requires_grad = True
			elif train_net == "modulator":
				for param in model.style_modulators.parameters():
					param.requires_grad = True


		optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

		print('The number of trainable parameters: {:d}'.format(sum(param.numel() for param in model.parameters() if param.requires_grad)))

		criterion = nn.BCEWithLogitsLoss()

		# Create template
		size = int(4200 * params['resize'])

		input_template = create_dist_mat(size=size)
		input_template = torch.Tensor(input_template).to(device)

		gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'], normalize=False)
		gt_template = torch.Tensor(gt_template).to(device)

		best_test_ADE = 99999999999999

		self.val_ADE = []
		self.val_FDE = []

		with_style = train_net == "modulator"
		print('Start training')
		for e in tqdm(range(params['num_epochs']), desc='Epoch'):
			train_ADE, train_FDE, train_loss = train(model, train_loader, train_images, e, obs_len, pred_len,
													 batch_size, params, gt_template, device,
													 input_template, optimizer, criterion, dataset_name, self.homo_mat, with_style=with_style)

			# For faster inference, we don't use TTST and CWS here, only for the test set evaluation
			val_ADE, val_FDE = evaluate(model, val_loader, val_images, num_goals, num_traj,
										obs_len=obs_len, batch_size=batch_size,
										device=device, input_template=input_template,
										waypoints=params['waypoints'], resize=params['resize'],
										temperature=params['temperature'], use_TTST=False,
										use_CWS=False, dataset_name=dataset_name,
										homo_mat=self.homo_mat, mode='val', with_style=with_style)

			print(f'Epoch {e}: 	Train (Top-1) ADE: {train_ADE:.2f} FDE: {train_FDE:.2f} 		Valid (Top-k) ADE: {val_ADE:.2f} FDE: {val_FDE:.2f}')
			self.val_ADE.append(val_ADE)
			self.val_FDE.append(val_FDE)

			if val_ADE < best_test_ADE:
				best_test_ADE = val_ADE
				best_state_dict = deepcopy(model.state_dict())

			if e % epochs_checkpoints == 0 and not fine_tune:
				torch.save(model.state_dict(), 'ckpts/' + experiment_name + f'_weights_epoch_{e}.pt')

			# early stop in case of clear overfitting
			if best_test_ADE < min(self.val_ADE[-5:]):
				print(f'Early stop at epoch {e}')
				break

		# Load best model
		model.load_state_dict(best_state_dict, strict=True)

		# # Save best model
		if not fine_tune:
			torch.save(best_state_dict, 'ckpts/' + experiment_name + '_weights.pt')

		return self.val_ADE, self.val_FDE

	def evaluate(self, data, params, image_path, batch_size=8, num_goals=20, num_traj=1, rounds=1, device=None, dataset_name=None, use_raw_data=False, with_style=False):
		"""
		Val function
		:param data: pd.df, val data
		:param params: dictionary with training hyperparameters
		:param image_path: str, filepath to val images
		:param batch_size: int, batch size
		:param num_goals: int, number of goals per trajectory, K_e in paper
		:param num_traj: int, number of trajectory per goal, K_a in paper
		:param rounds: int, number of epochs to evaluate
		:param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
		:return:
		"""

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			print('Working on GPU: ', torch.cuda.is_available())

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		elif dataset_name == 'ind':
			image_file_name = 'reference.png'
		elif dataset_name == 'eth':
			image_file_name = 'oracle.png'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		# ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
		if dataset_name == 'eth':
			self.homo_mat = {}
			for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
				self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
			seg_mask = True
		else:
			self.homo_mat = None
			seg_mask = False

		test_images = create_images_dict(data, image_path=image_path, image_file=image_file_name, use_raw_data=use_raw_data)

		test_dataset = SceneDataset(data, resize=params['resize'], total_len=total_len)
		test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=scene_collate)

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(test_images, factor=params['resize'], seg_mask=seg_mask)
		pad(test_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet architecture
		preprocess_image_for_segmentation(test_images, seg_mask=seg_mask)

		model = self.model.to(device)

		# Create template
		size = int(4200 * params['resize'])

		input_template = torch.Tensor(create_dist_mat(size=size)).to(device)

		self.eval_ADE = []
		self.eval_FDE = []

		print("TTST setting:", params['use_TTST'])
		print('Start testing')
		for e in tqdm(range(rounds), desc='Round'):
			test_ADE, test_FDE = evaluate(model, test_loader, test_images, num_goals, num_traj,
										  obs_len=obs_len, batch_size=batch_size,
										  device=device, input_template=input_template,
										  waypoints=params['waypoints'], resize=params['resize'],
										  temperature=params['temperature'], use_TTST=params['use_TTST'],
										  use_CWS=params['use_CWS'],
										  rel_thresh=params['rel_threshold'], CWS_params=params['CWS_params'],
										  dataset_name=dataset_name, homo_mat=self.homo_mat, mode='test', with_style=with_style)
			print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')

			self.eval_ADE.append(test_ADE)
			self.eval_FDE.append(test_FDE)

		ade = sum(self.eval_ADE) / len(self.eval_ADE)
		fde = sum(self.eval_FDE) / len(self.eval_FDE)
		print(f'\n\nAverage performance over {rounds} rounds: \nTest ADE: {ade} \nTest FDE: {fde}')
		return ade, fde


	def load(self, path):
		print(self.model.load_state_dict(torch.load(path), strict=False))

	def save(self, path):
		torch.save(self.model.state_dict(), path)
