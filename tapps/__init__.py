# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)

from .data.dataset_mappers.mask_former_panoptic_parts_dataset_mapper import (
    MaskFormerPanopticPartsDatasetMapper,
)

from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from .data.dataset_mappers.pascal_panoptic_parts_new_baseline_dataset_mapper import (
    PascalPanopticPartsNewBaselineDatasetMapper,
)

from .data.datasets.register_pascal_panoptic_parts import register_all_pascal_panoptic_parts
from .data.datasets.register_cityscapes_panoptic_parts import register_all_cityscapes_panoptic_parts
from .data.datasets.register_pascal_panoptic_parts_107 import register_all_pascal_panoptic_parts_107

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
