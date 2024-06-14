# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["PascalPanopticPartsNewBaselineDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class PascalPanopticPartsNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        ignore_label,
        meta,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.ignore_label = ignore_label
        self.meta = meta

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "meta": meta
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "part_seg_file_name" in dataset_dict:
            part_seg_gt = utils.read_image(dataset_dict.pop("part_seg_file_name")).astype("double")
            part_seg_gt = transforms.apply_segmentation(part_seg_gt)

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            masks_parts = []
            num_parts = []
            part_ids_list = []
            for segment_info in segments_info:
                mask = pan_seg_gt == segment_info["id"]
                if np.sum(mask) > 1:
                    class_id = segment_info["category_id"]
                    if not segment_info["iscrowd"]:
                        mask_parts_per_img = []
                        part_ids_list_per_img = []
                        if class_id in self.meta.train_sids_with_parts:
                            part_ids = self.meta.train_sids2train_pids[class_id]
                            for part_id in part_ids:
                                part_mask = part_seg_gt == part_id
                                inst_part_mask = np.logical_and(part_mask, mask)
                                mask_parts_per_img.append(inst_part_mask)
                                part_ids_list_per_img.append(part_id - 1)

                            if len(mask_parts_per_img) > 0:
                                mask_parts_per_img = BitMasks(
                                    torch.stack(
                                        [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_parts_per_img])
                                ).tensor
                            else:
                                mask_parts_per_img = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))

                            num_parts.append(len(part_ids_list_per_img))
                            part_ids_list.append(torch.from_numpy(np.ascontiguousarray(part_ids_list_per_img, dtype=np.longlong)))
                        else:
                            num_parts.append(0)
                            part_ids_list.append(torch.from_numpy(np.ascontiguousarray([], dtype=np.longlong)))
                            mask_parts_per_img = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                        masks_parts.append(mask_parts_per_img)
                        classes.append(class_id)
                        masks.append(mask)

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                instances.gt_masks_parts = list()
                instances.gt_num_parts = torch.zeros((0))
                instances.gt_part_ids = part_ids_list
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
                instances.gt_masks_parts = masks_parts
                instances.gt_num_parts = torch.from_numpy(np.ascontiguousarray(num_parts))
                instances.gt_part_ids = part_ids_list

            dataset_dict["instances"] = instances

        return dataset_dict
