# DTCD

We propose a segmentation-guided anomaly detection model based on dual-teacher contrastive distillation.

## Datasets

We use the MVTec AD dataset for experiments. To simulate anomalous image, the Describable Textures Dataset (DTD) is also adopted in our work. Users can run the **download_dataset.sh** script to download them directly.

```
./scripts/download_dataset.sh
```

## Installation

Please install the dependency packages using the following command by **pip**:

```
pip install -r requirements.txt
```

## **Foreground-Only Anomaly Generation and Foreground Segmentation Module:**

Please implement TokenCut and SAM according to the method described in our paper to generate object foreground masks for both the training and testing sets. （Alternatively, you may use the TokenCut and SAM scripts provided by us. For details, please refer to:[TokenCut-master(mvtec) - Google 云端硬盘](https://drive.google.com/drive/folders/17fZ0JLLIVb3demmGxQyzFy-733zRBUFk?hl=zh-cn)）.



## Training and Testing

To get started, users can run the following command to train the model on all categories of MVTec AD dataset:

```
python train_old_front-trueRotate123.py --gpu_id 0 --num_workers 16
```

Users can also customize some default training parameters by resetting arguments like `--bs`, `--lr_DeST`, `--lr_res`, `--lr_seghead`, `--steps`, `--DeST_steps`, `--eval_per_steps`, `--log_per_steps`, `--gamma` and `--T`.

To specify the training categories and the corresponding data augmentation strategies, please add the argument `--custom_training_category` and then add the categories after the arguments `--no_rotation_category`, `--slight_rotation_category` and `--rotation_category`. For example, to train the `screw` category and the `tile` category with no data augmentation strategy, just run the following command:

```
python train_old_front-trueRotate123.py --gpu_id 0 --num_workers 16 --custom_training_category --no_rotation_category screw tile
```

To test the performance of the model, users can run the following command:

```
python eval_new_new.py --gpu_id 0 --num_workers 16
```

After validation is completed, please enable the foreground segmentation module for the specified categories.

```
python eval_new_front.py --gpu_id 0 --num_workers 16
```



## Pretrained Checkpoints

Download pretrained checkpoints:[DTCD - Google 云端硬盘](https://drive.google.com/drive/folders/1_r8KtKfPidLVLMZXcM-Q3m51fBX_z0Aa?hl=zh-cn)

## Citation
