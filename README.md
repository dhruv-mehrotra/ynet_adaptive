# Y-net adaptive (Causal Motion Forecasting)


This is an addition to the [official implementation](https://github.com/vita-epfl/causalmotion) for the paper

**Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective**
<br>
<a href="https://sites.google.com/view/yuejiangliu">Yuejiang Liu</a>,
<a href="https://www.riccardocadei.com">Riccardo Cadei</a>,
<a href="https://people.epfl.ch/jonas.schweizer/?lang=en">Jonas Schweizer</a>,
<a href="https://sherwinbahmani.github.io">Sherwin Bahmani</a>,
<a href="https://people.epfl.ch/alexandre.alahi/?lang=en/">Alexandre Alahi</a>
<br>
École Polytechnique Fédérale de Lausanne (EPFL)

Links: **[`Arxiv 11/2021`](https://arxiv.org/abs/2111.14820) | [`Video (7 min)`](https://drive.google.com/file/d/1Uo0Y0eHq4vI7wOxya4mJlxbAe3U4kMx6/view) | [`Adaptive Y-net`](https://github.com/sherwinbahmani/ynet_adaptive)**
<br>
*Under review. Abbreviated version at NeurIPS DistShift, 2021.*

TL;DR: incorporate causal invariance and structure into the design and training of motion forecasting models
* causal formalism of motion forecasting with three groups of latent variables
* causal (invariant) representations to suppress spurious features and promote robust generalization
* causal (modular) structure to approximate a sparse causal graph and facilitate efficient adaptation

<p align="left">
  <img src="overview.png" width="800">
</p>

If you find this code useful for your research, please cite our paper:

```bibtex
@article{liu2021causalmotion,
  title={Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective},
  author={Liu, Yuejiang and Cadei, Riccardo and Schweizer, Jonas and Bahmani, Sherwin and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2111.14820},
  year={2021}
}
```
### Original Code

Based on the original [Y-net](https://arxiv.org/pdf/2012.01526.pdf) [repository](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet)

### Setup

Environments

```
pip install --upgrade pip
pip install -r requirements.txt
# pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Dataset
Get segmentation masks for SDD from original Y-net authors
```
pip install gdown && gdown https://drive.google.com/uc?id=1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa
cd -rf ynet_additional_files/* ./
```

Download dataset https://www.kaggle.com/aryashah2k/stanford-drone-dataset and unzip to a directory dataset_raw
Then run the preprocessing of the dataset to create a filtered dataset dataset_filter
```
python utils/dataset.py --data_raw {Path to dataset_raw directory} --data_filter {Path to output dataset_filter directory} --labels {Filter dataset based on agent types}
```

### Setup
Train Baseline

```
bash run_train.sh
```

Zero-shot Evaluation

```
bash run_eval.sh
```

Fine-tuning / Few-shot adaptation

```
bash run_fine_tune.sh
```
