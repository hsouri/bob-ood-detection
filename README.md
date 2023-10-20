# BoB-OOD-Detection

This repository is the official implementation of <strong>OOD Object Detection</strong> task in the [*Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks*](https://github.com/hsouri/Battle-of-the-Backbones).

:pushpin: Our implementation and instructions are based on [mmdetection](https://github.com/open-mmlab/mmdetection)

## Installation

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

**Step 3.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full==1.7.0
```

**Step 4.** Install BoB-OOD-Detection.

```shell
git clone https://github.com/hsouri/bob-ood-detection.git
cd bob-ood-detection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Step 5.** Setup Datasets.

Download the <a href="https://fcav.engin.umich.edu/projects/driving-in-the-matrix">Sim10k</a> dataset and run the following command to process annotations.

```
python dataset_utils/sim10k_voc2coco_format.py \
    --sim10k_path <path-to-sim10k-folder> \
    --img-dir <path-to-sim10k-images> \
    --gt-dir <path-to-sim10k-annotations> \
    --out-dir <path-to-store-processed-annotations>
```

Download the <a href="https://www.cityscapes-dataset.com/downloads/">Cityscapes</a> dataset.

Once processed, update the path to individual datasets in the experiment configs at [configs/bob_sim2real](https://github.com/hsouri/bob-ood-detection/tree/main/configs/bob_sim2real).

If required, please refer to [Get Started](https://github.com/hsouri/bob-detection/blob/master/docs/en/get_started.md), [Dataset Prepare](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html?highlight=dataset), and [Dataset Download](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download) for more detailed instructions.

## Usage

The config files for all experiments in <strong>Battle of the Backbones (BoB)</strong> can be found [configs/bob_sim2real](https://github.com/hsouri/bob-detection/tree/master/configs/bob_sim2real).

To train a detector with the existing configs, run:

```shell
bash ./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>
```
