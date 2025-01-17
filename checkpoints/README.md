# Download pre-trained model weights

## TAPPS model weights
[Here](../MODELS.md), we list the different models that we release, and provide a download link.


## COCO pre-trained model weights
To initialize a model with COCO (panoptic) pre-trained weights, as done in our work, follow these steps:

1. Identify the backbone architecture of the model you wish to train (e.g., ResNet-50 or Swin-B)
2. For this backbone, download the model weights provided in the [original Mask2Former repository](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md#panoptic-segmentation), trained for COCO panoptic segmentation.
3. Place these model weights in the `checkpoints` directory, following this structure:

    ```
    checkpoints/
        maskformer2_R50_bs16_50ep/
            model_final_94dc52.pkl
        maskformer2_swin_base_IN21k_384_bs16_50ep/
            model_final_54b88a.pkl
    ```

Then, you can simply run the training code following [the instructions provided here](../README.md#training). In the default configs, the path to the COCO pre-trained weights is already provided.