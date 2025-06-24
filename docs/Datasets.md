# Datasets


## Supported Datasets

| Class         | Task   |Train/Test | Description       |
| :------------- | :----------:| :----------:    | :----------:   |
| [REDSDataset](../basicsr/data/reds_recurrent_dataset.py) | Video Super-Resolution | Train or Test|REDS training dataset |
| [Vimeo90KDataset](../basicsr/data/vimeo90k_dataset.py) | Video Super-Resolution |Train| Vimeo90K training dataset|
| [VideoTestDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution | Test|Base video test dataset, supporting Vid4, Vimeo90K testing datasets|
| [VideoTestVimeo90KDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution |Test| Inherit `VideoTestDataset`, Vimeo90K testing dataset|
| [RealVSRDataset](../basicsr/data/real_recurrent_dataset.py) | Video Super-Resolution |Train or Test| RealVSR testing dataset|