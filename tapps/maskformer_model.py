# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import os
from PIL import Image
import numpy as np
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.transformer_decoder.part_decoder import PartDecoder

sys.path.append("utils/panoptic_parts")
from panoptic_parts.specs.eval_spec import PartPQEvalSpec

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        parts_on: bool,
        test_topk_per_image: int,
        save_dir: str,
        save_predictions: bool,
        num_part_classes: int,
        parts_conf_threshold: float,
        eval_spec,
        cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.parts_on = parts_on
        self.test_topk_per_image = test_topk_per_image

        # Extra
        self.save_predictions = save_predictions
        self.save_dir = save_dir
        self.num_part_classes = num_part_classes
        self.parts_conf_threshold = parts_conf_threshold
        self.eval_spec = eval_spec

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        if self.eval_spec is not None:
            sid_total = eval_spec.eval_sid_total
            trainid2datasetid = dict()
            for i, sid in enumerate(sid_total):
                trainid2datasetid[i] = sid
            self.trainid2datasetid = trainid2datasetid
            self.partsegid2pid = self._prepare_part_mappings()
        else:
            self.trainid2datasetid = None

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        parts_weight = cfg.MODEL.MASK_FORMER.LOSS_WEIGHT_PARTS
        panoptic_weight = cfg.MODEL.MASK_FORMER.LOSS_WEIGHT_PANOPTIC

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight * panoptic_weight,
                       "loss_mask": mask_weight * panoptic_weight,
                       "loss_dice": dice_weight * panoptic_weight}

        if cfg.MODEL.MASK_FORMER.PARTS_ON:
            weight_dict.update({"loss_mask_parts": mask_weight * parts_weight})
            weight_dict.update({"loss_dice_parts": dice_weight * parts_weight})

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])

        if cfg.MODEL.MASK_FORMER.PARTS_ON:
            eval_spec = PartPQEvalSpec(cfg.PPS_EVAL_SPEC)
        else:
            eval_spec = None

        if cfg.MODEL.MASK_FORMER.PARTS_ON:
            parts_decoder = PartDecoder(num_part_classes=cfg.MODEL.MASK_FORMER.NUM_PART_CLASSES,
                                        input_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
                                        hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
                                        mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
                                        )
        else:
            parts_decoder = None

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            parts_on=cfg.MODEL.MASK_FORMER.PARTS_ON,
            parts_decoder=parts_decoder,
        )

        return {
            "cfg": cfg,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "parts_on": cfg.MODEL.MASK_FORMER.TEST.PARTS_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "save_dir": cfg.SAVE_DIR,
            "save_predictions": cfg.SAVE_PREDICTIONS,
            "num_part_classes": cfg.MODEL.MASK_FORMER.NUM_PART_CLASSES,
            "parts_conf_threshold": cfg.MODEL.MASK_FORMER.PARTS_CONF_THRESHOLD,
            "eval_spec": eval_spec,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for b, (mask_cls_result, mask_pred_result, input_per_image, image_size) in enumerate(zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                    if self.save_predictions:
                        save_dir_semantic = os.path.join(self.save_dir, "semantic")
                        sem_preds = torch.argmax(r, dim=0)
                        pred_img = Image.fromarray(sem_preds.detach().cpu().numpy().astype(np.uint8))
                        pred_img.save(os.path.join(save_dir_semantic, str(batched_inputs[0]['image_id']) + ".png"))

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result,
                                                                            mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r[0:2]

                    if self.save_predictions:
                        panoptic_pred = panoptic_r[0]
                        segments_info = panoptic_r[1]
                        segm_info = segments_info.copy()

                        pred_np = panoptic_pred.detach().cpu().numpy()
                        pred_img = Image.fromarray(pred_np.astype(np.uint8))
                        save_dir_panoptic = os.path.join(self.save_dir, "panoptic")

                        pred_img.save(os.path.join(save_dir_panoptic, str(batched_inputs[0]['image_id']) + ".png"))
                        with open(os.path.join(save_dir_panoptic, str(batched_inputs[0]['image_id']) + ".json"),
                                  'w') as fp:
                            json.dump(segm_info, fp)

                    if self.parts_on:
                        partseg_pred, pps_pred = retry_if_cuda_oom(self.pps_inference)(panoptic_r,
                                                                                         outputs["mask_features"],
                                                                                         mask_cls_result,
                                                                                         outputs["decoder_output"],
                                                                                         b,
                                                                                         images.tensor.shape,
                                                                                         image_size,
                                                                                         height,
                                                                                         width)

                        processed_results[-1]["parts"] = partseg_pred
                        processed_results[-1]["pps"] = pps_pred

                        if self.save_predictions:
                            # Uncomment this if you wish to store the part segmentation predictions (also see train_net.py)
                            # pred_np = partseg_pred.detach().cpu().numpy()
                            # pred_img = Image.fromarray(pred_np.astype(np.uint8))
                            # save_dir_parts = os.path.join(self.save_dir, "parts")
                            # pred_img.save(os.path.join(save_dir_parts, str(batched_inputs[0]['image_id']) + ".png"))

                            pps_pred_total = torch.stack(pps_pred, dim=-1)
                            pps_pred_np = pps_pred_total.detach().cpu().numpy()
                            pps_pred_img = Image.fromarray(pps_pred_np.astype(np.uint8))
                            save_dir_pps = os.path.join(self.save_dir, "pps")
                            pps_pred_img.save(os.path.join(save_dir_pps, str(batched_inputs[0]['image_id']) + ".png"))

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            target_dict = {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }

            if self.parts_on:
                gt_masks_parts_per_pan = targets_per_image.gt_masks_parts
                gt_masks_parts_per_pan_new = list()
                for masks_parts in gt_masks_parts_per_pan:
                    padded_masks = torch.zeros((masks_parts.shape[0], h_pad, w_pad),
                                               dtype=masks_parts.dtype,
                                               device=gt_masks.device)
                    padded_masks[:, : masks_parts.shape[1], : masks_parts.shape[2]] = masks_parts
                    gt_masks_parts_per_pan_new.append(padded_masks)
                target_dict["masks_parts"] = gt_masks_parts_per_pan_new
                target_dict["num_masks_parts"] = targets_per_image.gt_num_parts
                target_dict["part_ids"] = [part_ids.to(device=gt_masks.device) for part_ids in targets_per_image.gt_part_ids]

            new_targets.append(target_dict)

        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        mask_cls_softmax = F.softmax(mask_cls, dim=-1)
        scores, labels = mask_cls_softmax.max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        query_ids = torch.arange(0, scores.shape[0], device=mask_pred.device)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_query_ids = query_ids[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []
        part_related_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info, part_related_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

                    part_related_info.append(
                        {
                            "panoptic_id": current_segment_id,
                            "query_id": cur_query_ids[k],
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info, part_related_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def _prepare_part_mappings(self, void=255):
        sid_pid2part_seg_id = self.eval_spec.eval_sid_pid2eval_pid_flat

        part_seg_id2pid = np.full([256], fill_value=void, dtype=np.int32)
        for sid_pid, part_seg_id in zip(sid_pid2part_seg_id.keys(), sid_pid2part_seg_id.values()):
            pid = sid_pid % 100
            if part_seg_id2pid[part_seg_id] == 255:
                part_seg_id2pid[part_seg_id] = pid
            else:
                if part_seg_id2pid[part_seg_id] != pid:
                    raise NotImplementedError("This codebase can currently only map part segmentation ids to pids (necessary for PartPQ eval) if there is a unique part_seg_id -> pid mapping. To still evaluate PartPQ, separately save the part and panoptic segmentation predictions and merge to PPS via code on https://github.com/pmeletis/panoptic_parts/.")

        return part_seg_id2pid

    def pps_inference(self, panoptic_r, mask_features, mask_cls_result, queries, b, images_shape,
                      image_size, height, width):
        # 1. Get the panoptic segments (after merging), their classes and the corresponding query ids
        part_related_info = panoptic_r[2]

        gather_ids = []
        num_parts_per_query = []
        part_ids_per_query = []

        sem_id_map = torch.full([height,width],
                                255,
                                dtype=torch.long,
                                device=mask_cls_result.device)
        inst_id_map = torch.full([height,width],
                                 255,
                                 dtype=torch.long,
                                 device=mask_cls_result.device)

        # 2. Identify queries with a class prediction that requires parts
        for p_info in part_related_info:
            category_id = p_info['category_id']
            # Retrieve the queries for classes with parts
            if category_id in self.metadata.train_sids_with_parts:
                gather_ids.append(p_info['query_id'])
                num_parts_per_query.append(len(self.metadata.train_sids2train_pids[category_id]))
                part_ids_per_query.append(
                    torch.from_numpy(
                        np.ascontiguousarray(self.metadata.train_sids2train_pids[category_id]) - 1).to(
                        mask_features.device))

        if len(gather_ids) > 0:
            gather_ids = torch.stack(gather_ids)

            # Gather the queries and the features for this img
            queries_selected = queries[b][gather_ids].unsqueeze(0)
            mask_features_img = mask_features[b].unsqueeze(0)

            # 3. Pass the queries through the 'criteríon' top-down part heads
            masks_parts = self.criterion.parts_decoder(queries_selected,
                                                       [num_parts_per_query],
                                                       mask_features_img,
                                                       [part_ids_per_query])

            # mask_parts shape [sum(num_parts_per_query), H, W]
            # Upsample them to the original input dimensions
            masks_parts = F.interpolate(
                masks_parts.unsqueeze(0),
                size=(images_shape[-2], images_shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            masks_parts = retry_if_cuda_oom(sem_seg_postprocess)(
                masks_parts, image_size, height, width
            )
            masks_parts = torch.sigmoid(masks_parts)

            # First create a partseg tensor of shape [self.num_part_classes, H, W] (including bg class)
            partseg_map = torch.full([self.num_part_classes + 1,
                                      height,
                                      width],
                                     fill_value=self.parts_conf_threshold,
                                     dtype=torch.float,
                                     device=mask_cls_result.device)
            panoptic_pred = panoptic_r[0]

            part_count = 0
            inst_count = 1

            for p_info in part_related_info:
                category_id = p_info['category_id']
                panoptic_id = p_info['panoptic_id']
                # Get the panoptic mask
                mask = panoptic_pred == panoptic_id
                sem_id_map[mask] = self.trainid2datasetid[category_id]
                if category_id in self.metadata.thing_dataset_id_to_contiguous_id.values():
                    inst_id = inst_count
                    inst_count += 1
                else:
                    inst_id = 1
                inst_id_map[mask] = inst_id
                # Identify each pan mask that requires parts
                if category_id in self.metadata.train_sids_with_parts:
                    part_ids = self.metadata.train_sids2train_pids[category_id]
                    num_part_ids = len(part_ids)
                    # For the compatible part ids, and within the panoptic mask, store the part logits
                    bg_count = 0
                    for i, part_id in enumerate(part_ids):
                        partseg_map[part_id][mask] = masks_parts[part_count + bg_count + i][mask]
                    part_count += num_part_ids + bg_count

            # Take the argmax to make the final predictions
            partseg_pred = torch.argmax(partseg_map, dim=0)

            if inst_count > 256:
                raise NotImplementedError("Instance ID is above 255, which is currently not supported in the data format desired by panoptic_parts.")

        else:
            partseg_pred = torch.zeros([height,
                                        width],
                                       dtype=torch.long,
                                       device=mask_cls_result.device)

            panoptic_pred = panoptic_r[0]
            inst_count = 1
            for p_info in part_related_info:
                category_id = p_info['category_id']
                panoptic_id = p_info['panoptic_id']
                mask = panoptic_pred == panoptic_id
                sem_id_map[mask] = self.trainid2datasetid[category_id]
                if category_id in self.metadata.thing_dataset_id_to_contiguous_id.values():
                    inst_id = inst_count
                    inst_count += 1
                else:
                    inst_id = 1
                inst_id_map[mask] = inst_id

        partsegid2pid_mapping = torch.from_numpy(self.partsegid2pid).to(dtype=torch.long, device=partseg_pred.device)
        part_id_map = partsegid2pid_mapping[partseg_pred]

        return partseg_pred, [sem_id_map, inst_id_map, part_id_map]