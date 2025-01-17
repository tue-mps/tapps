# TAPPS: Task-aligned Part-aware Panoptic Segmentation (CVPR 2024)
## [[Project page](https://tue-mps.github.io/tapps/)] [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/de_Geus_Task-aligned_Part-aware_Panoptic_Segmentation_through_Joint_Object-Part_Representations_CVPR_2024_paper.pdf)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/task-aligned-part-aware-panoptic-segmentation-1/part-aware-panoptic-segmentation-on)](https://paperswithcode.com/sota/part-aware-panoptic-segmentation-on?p=task-aligned-part-aware-panoptic-segmentation-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/task-aligned-part-aware-panoptic-segmentation-1/part-aware-panoptic-segmentation-on-pascal)](https://paperswithcode.com/sota/part-aware-panoptic-segmentation-on-pascal?p=task-aligned-part-aware-panoptic-segmentation-1)

Code for 'Task-aligned Part-aware Panoptic Segmentation through Joint Object-Part Representations', Daan de Geus and Gijs Dubbelman, CVPR 2024.

## Installation

* Follow the [installation instructions](INSTALL.md).

## Getting Started

* [Prepare the datasets for TAPPS](datasets/README.md).
* To use COCO panoptic pre-training, download the checkpoints following [these instructions](checkpoints/README.md).

## Training
To train a model, you need a configuration file. We provide [default configuration files](MODELS.md) for the models presented in our work. Our configs are designed to be trained on 4 GPUs, using a different number of GPUs likely requires changes to the learning rate.

```bash
python train_net.py --num-gpus 4 \
                    --config-file /PATH/TO/CONFIG/FILE.yaml 
```

Example:

```bash
python train_net.py --num-gpus 4 \
                    --config-file configs/pascal/pps/tapps_pascal_r50_cocoinit.yaml
```

## Evaluation

Evaluating a model on the PartPQ metric requires two steps: (1) making and saving the predictions, (2) evaluating the predictions and calculating the PartPQ.

### 1. Making predictions and saving them
This step requires a configuration file and the model weights. We provide [default configuration files and weights](MODELS.md) for the models presented in our work. You also need to define a directory where you wish to store the predictions.

```bash
python train_net.py --num-gpus 4 \
                    --config-file /PATH/TO/CONFIG/FILE.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /PATH/TO/MODEL/WEIGHTS.bin \
                    SAVE_PREDICTIONS True \
                    SAVE_DIR /PATH/WHERE/PREDICTIONS/ARE/STORED/
```

Example:

```bash
python train_net.py --num-gpus 4 \
                    --config-file configs/pascal/pps/tapps_pascal_r50_cocoinit.yaml \
                    --eval-only \
                    MODEL.WEIGHTS checkpoints/tapps_pascal_r50_cocoinit.bin \
                    SAVE_PREDICTIONS True \
                    SAVE_DIR predictions/
```


### 2. Evaluating the predictions and calculating the PartPQ
Provide the path where the predictions are stored and specify the dataset.

```bash
python eval/eval_partpq.py --save_dir /PATH/WHERE/PREDICTIONS/ARE/STORED/ \
                           --dataset DATASET_NAME  # 'pascal', 'cityscapes' or 'pascal107'
```

Example:

```bash
python eval/eval_partpq.py --save_dir predictions/ \
                           --dataset pascal 
```

## Visualization
To visualize the part-aware panoptic segmentation predictions that you stored (see Evaluation step 1), run:

```bash
python eval/visualize_pps.py --pred_dir /PATH/WHERE/PREDICTIONS/ARE/STORED/ \
                             --save_dir /PATH/WHERE/VISUALIZATIONS/WILL/BE/STORED/ \
                             --dataset DATASET_NAME  # 'pascal', 'cityscapes' or 'pascal107'
```

## Single-image inference
To run inference on a single image and store the visualized prediction, run:


```bash
python inference_single_img.py --config /PATH/TO/CONFIG/FILE.yaml \
                               --model_weights /PATH/TO/MODEL/WEIGHTS.bin \
                               --image /PATH/TO/IMAGE/FILE.jpg \  # can also be another image format
                               --save_dir /PATH/WHERE/PREDICTION/WILL/BE/STORED/
```

For example:

```bash
python inference_single_img.py --config configs/pascal/pps/tapps_pascal_swinb_cocoinit.yaml \
                               --model_weights checkpoints/tapps_pascal_swinb_cocoinit.bin \
                               --image data/pascal/JPEGImages/2010_005252.jpg \ 
                               --save_dir predictions/
```


## Models
Check [this page](MODELS.md) for trained models and associated config files.


## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This code builds upon the [official Mask2Former code](https://github.com/facebookresearch/Mask2Former/). The majority of Mask2Former is licensed under an [MIT License](LICENSE_MASK2FORMER). However, portions of the Mask2Former project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

This code also uses the [panoptic_parts](https://github.com/pmeletis/panoptic_parts/) code, which is licensed under the [Apache-2.0 license](https://github.com/pmeletis/panoptic_parts/blob/master/LICENSE).

The remaining code, specifically added for TAPPS, is licensed under an [MIT license](LICENSE).

## <a name="Citing"></a>Citing us

Please consider citing our work if it is useful for your research.

```BibTeX
@inproceedings{degeus2024tapps,
  title={{Task-aligned Part-aware Panoptic Segmentation through Joint Object-Part Representations}},
  author={{de Geus}, Daan and Dubbelman, Gijs},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

If you use the part-aware panoptic segmentation task in your work, consider citing:

```BibTeX
@inproceedings{degeus2021pps,
  title = {{Part-aware Panoptic Segmentation}},
  author = {{de Geus}, Daan and Meletis, Panagiotis and Lu, Chenyang and Wen, Xiaoxiao and Dubbelman, Gijs},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021},
}
```


## Acknowledgement

This project is built on top of the official code of [Mask2Former](https://github.com/facebookresearch/Mask2Former/), we thank the authors for their great work and useful code.
