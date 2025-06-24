# Dataset Preparation

[English](DatasetPreparation.md) 


## Data Storage Format

At present, there are three types of data storage formats supported:

1. Store in `hard disk` directly in the format of images / video frames.
1. Make [LMDB](https://lmdb.readthedocs.io/en/release/), which could accelerate the IO and decompression speed during training.

#### How to Use

At present, we can modify the configuration yaml file to support different data storage formats. Taking [PairedImageDataset](../basicsr/data/paired_image_dataset.py) as an example, we can modify the yaml file according to different requirements.

1. Directly read disk data.

    ```yaml
    type: VideoTestDataset
    dataroot_gt: ./train_sharp
    dataroot_lq: ./train_sharp_bicubic/X4/
    io_backend:
      type: disk
    ```

1. Use LMDB.
We need to make LMDB before using it. Please refer to [LMDB description](#LMDB-Description). Note that we add meta information to the original LMDB, and the specific binary contents are also different. Therefore, LMDB from other sources can not be used directly.

    ```yaml
    type: REDSDataset
    dataroot_gt: /cluster/work/cvl/videosr/REDS/train_sharp_with_val.lmdb 
    dataroot_lq: /cluster/work/cvl/videosr/REDS/train_sharp_bicubic_with_val.lmdb 
    io_backend:
      type: lmdb
    ```



## Video Super-Resolution

It is recommended to symlink the dataset root to `datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### REDS

[Official website](https://seungjunnah.github.io/Datasets/reds.html).<br>
We regroup the training and validation dataset into one folder. The original training dataset has 240 clips from 000 to 239. And we  rename the validation clips from 240 to 269.

**Validation Partition**

The official validation partition and that used in EDVR for competition are different:

| name | clips | total number |
|:----------:|:----------:|:----------:|
| REDSOfficial | [240, 269] | 30 clips |
| REDS4 | 000, 011, 015, 020 clips from the *original training set* | 4 clips |

All the left clips are used for training. Note that it it not required to explicitly separate the training and validation datasets; and the dataloader does that.

**Preparation Steps**

1. Download the datasets from the [official website](https://seungjunnah.github.io/Datasets/reds.html).
1. Regroup the training and validation datasets: `python scripts/data_preparation/regroup_reds_dataset.py`
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_reds` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_reds_dataset.py`.
Remember to modify the paths and configurations accordingly.

### Vimeo90K

[Official webpage](http://toflow.csail.mit.edu/)

1. Download the dataset: [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).This is the Ground-Truth (GT). There is a `sep_trainlist.txt` file listing the training samples in the download zip file.
1. Generate the low-resolution images (TODO)
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_vimeo90k` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_vimeo90k_dataset.py`.
Remember to modify the paths and configurations accordingly.

### RealVSR Dataset

The dataset is hosted on [Google Drive](https://drive.google.com/drive/folders/1-8MvMEYMOeOE713DjI7TJKyRE-LnrM3Y?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1rBIGo5xrY2VtpoUF2gf_HA) (code: 43ph). Some example scenes are shown below.


The structure of the dataset is illustrated below.

| File                     | Description                                 |
| ------------------------ |:-------------------------------------------:|
| GT.zip                   | All ground truth sequences in RGB format    |
| LQ.zip                   | All low quality sequences in RGB format     |
| GT_YCbCr.zip             | All ground truth sequences in YCbCr format  |
| LQ_YCbCr.zip             | All low quality sequences in YCbCr format   |
| GT_test.zip              | Ground truth test sequences in RGB format   |
| LQ_test.zip              | Low Quality test sequences in RGB format    |
| GT_YCbCr_test.zip        | Ground truth test sequences in YCbCr format |
| LQ_YCbCr_test.zip        | Low Quality test sequences in YCbCr format  |
| videos.zip               | Original videos (> 500 LR-HR pairs here)    |