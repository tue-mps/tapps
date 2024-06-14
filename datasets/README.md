# Prepare Datasets for TAPPS

TAPPS has builtin support for two datasets: Cityscapes Panoptic Parts (Cityscapes-PP) and Pascal Panoptic Parts (Pascal-PP)
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below.
```
$DETECTRON2_DATASETS/
  cityscapes/
  pascal/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

## Expected dataset structure for [Cityscapes-PP](https://github.com/pmeletis/panoptic_parts):

First download the [Cityscapes dataset](https://cityscapes-dataset.com/downloads/) and put the data in the `cityscapes` directory. Download `gtFine_trainvaltest.zip`, `leftImg8bit_trainvaltest.zip`, and `gtFinePanopticParts.zip`. Structure it as below:

```
cityscapes/
  gtFine/
    train/
      aachen/
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
  gtFinePanopticParts/
    train/
    val/  
```
In any directory, clone cityscapesScripts by:
```bash
git clone https://github.com/mcordts/cityscapesScripts.git
```

To create labelTrainIds.png, first prepare the above structure, then run cityscapesScripts with:
```bash
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

To generate Cityscapes panoptic dataset, run cityscapesScripts with:
```bash
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```

To prepare the part segmentation files necessary for training, run:
```bash
python datasets/prepare_cityscapes_pp.py
```

After doing this, the data should be in the following structure:
```
cityscapes/
  gtFine/
    train/
      aachen/
      ...
    val/
    test/
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
  gtFinePanopticParts/
    train/
    val/
  gtFineParts/
    train/
    val/
  images_val.json
  images_train.json  
```

## Expected dataset structure for [Pascal-PP](https://github.com/pmeletis/panoptic_parts):

Download the [Pascal-PP labels](https://github.com/pmeletis/panoptic_parts) and the [Pascal VOC 2010 images](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/). Organize the data in the following structure:

```
pascal/
  JPEGImages/    # From VOC2010
  labels/        # From pascal_panoptic_parts_v2.0
    training/
    validation/
```

To generate the panoptic, semantic and part segmentation annotations and split the images into training and validation splits, run:
```bash
python datasets/prepare_pascal_pp.py
```

Afterwards, the data should have the following structure:
```
pascal/
  images/
    training/
    validation/
  labels/
    training/
    validation/
  panoptic/
    training/
    validation/
    panoptic_training.json
    panoptic_validation.json
  semantic/
    training/
    validation/
  parts/
    training/
    validation/
  images_training.json
  images_validation.json
```

Note: if you wish to use the Pascal-PP-107 labels (instead of the default Pascal-PP-57), also run:
```bash
python datasets/prepare_pascal_pp_107.py
```