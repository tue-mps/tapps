# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
from .augmentations import AugInput

__all__ = ["MaskFormerPanopticPartsDatasetMapper"]


class MaskFormerPanopticPartsDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

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
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        meta,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility,
            meta=meta,
        )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # part segmentation
        if "part_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            part_seg_gt = utils.read_image(dataset_dict.pop("part_seg_file_name")).astype("double")
        else:
            part_seg_gt = None

        aug_input = AugInput(image, sem_seg=sem_seg_gt, part_seg=part_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        if sem_seg_gt is not None:
            sem_seg_gt = aug_input.sem_seg
        if part_seg_gt is not None:
            part_seg_gt = aug_input.part_seg

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        if part_seg_gt is not None:
            part_seg_gt = torch.as_tensor(part_seg_gt.astype("long"))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            if part_seg_gt is not None:
                part_seg_gt = F.pad(part_seg_gt, padding_size, value=self.ignore_label).contiguous()
            pan_seg_gt = F.pad(
                pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if part_seg_gt is not None:
            dataset_dict["part_seg"] = part_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Pemantic segmentation dataset should not have 'annotations'.")

        # Prepare panoptic information with accompanying part information
        pan_seg_gt = pan_seg_gt.numpy()
        if part_seg_gt is not None:
            part_seg_gt = part_seg_gt.numpy()
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
                        part_ids_list.append(
                            torch.from_numpy(np.ascontiguousarray(part_ids_list_per_img, dtype=np.longlong)))
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
            instances.gt_masks_parts = list()
            instances.gt_num_parts = torch.zeros((0))
            instances.gt_part_ids = part_ids_list
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_masks_parts = masks_parts
            instances.gt_num_parts = torch.from_numpy(np.ascontiguousarray(num_parts))
            instances.gt_part_ids = part_ids_list

        dataset_dict["instances"] = instances

        return dataset_dict
