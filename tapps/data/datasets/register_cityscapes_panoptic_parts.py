# Copyright (c) Facebook, Inc. and its affiliates.
import os
import sys
import json
import logging
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager

sys.path.append("utils/panoptic_parts")
from panoptic_parts.specs.eval_spec import PartPQEvalSpec

logger = logging.getLogger(__name__)


def get_cityscapes_panoptic_parts_files(image_dir, gt_dir, json_info):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    image_dict = {}
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "_leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = os.path.basename(basename)[: -len(suffix)]

            image_dict[basename] = image_file

    for ann in json_info["annotations"]:
        image_file = image_dict.get(ann["image_id"], None)
        assert image_file is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]

        files.append((image_file, label_file, segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files

def load_cityscapes_panoptic_parts(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
    with open(gt_json) as f:
        json_info = json.load(f)
    files = get_cityscapes_panoptic_parts_files(image_dir, gt_dir, json_info)
    ret = []
    for image_file, label_file, segments_info in files:
        part_label_file = (
            image_file.replace("leftImg8bit", "gtFineParts").replace("gtFineParts.png", "gtFinePanopticParts.png")
        )
        sem_label_file = (
            image_file.replace("leftImg8bit", "gtFine").split(".")[0] + "_labelTrainIds.png"
        )
        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
                ),
                "sem_seg_file_name": sem_label_file,
                "part_seg_file_name": part_label_file,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    assert PathManager.isfile(
        ret[0]["part_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    return ret


_RAW_CITYSCAPES_PANOPTIC_SPLITS = {
    "cityscapes_panoptic_parts_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/gtFine/cityscapes_panoptic_train",
        "cityscapes/gtFine/cityscapes_panoptic_train.json",
    ),
    "cityscapes_panoptic_parts_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/gtFine/cityscapes_panoptic_val",
        "cityscapes/gtFine/cityscapes_panoptic_val.json",
    ),
}


def _prepare_mappings(sid_pid2part_seg_label, void):
  # Get the maximum amount of part_seg labels
  num_part_seg_labels = np.max(
    list(sid_pid2part_seg_label.values()))

  sids2part_seg_ids = dict()
  for class_key in sid_pid2part_seg_label.keys():
    class_id = class_key // 100
    if class_id in sids2part_seg_ids.keys():
      if sid_pid2part_seg_label[class_key] not in sids2part_seg_ids[class_id]:
        sids2part_seg_ids[class_id].append(sid_pid2part_seg_label[class_key])
      else:
        raise ValueError(
          'A part seg id can only be shared between different semantic classes, not within a single semantic class.')
    else:
      sids2part_seg_ids[class_id] = [sid_pid2part_seg_label[class_key]]

  sids2pids_eval = dict()
  for class_key in sid_pid2part_seg_label.keys():
    class_id = class_key // 100
    if class_id in sids2pids_eval.keys():
      if class_key % 100 not in sids2pids_eval[class_id]:
        sids2pids_eval[class_id].append(class_key % 100)
    else:
      sids2pids_eval[class_id] = [class_key % 100]

  part_seg_ids2eval_pids_per_sid = dict()
  for class_key in sids2part_seg_ids.keys():
    tmp = np.ones(num_part_seg_labels + 1, np.uint8) * void
    tmp[sids2part_seg_ids[class_key]] = sids2pids_eval[class_key]
    part_seg_ids2eval_pids_per_sid[class_key] = tmp

  return sids2part_seg_ids, part_seg_ids2eval_pids_per_sid


def register_all_cityscapes_panoptic_parts(cfg):
    root = os.getenv("DETECTRON2_DATASETS", "datasets")

    meta = {}

    eval_spec = PartPQEvalSpec(cfg.PPS_EVAL_SPEC)

    # Load panoptic seg meta
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    # Get mappings from scene-level to part-level
    sid_pid2part_seg_label = eval_spec.eval_sid_pid2eval_pid_flat
    sids2part_seg_ids, part_seg_ids2eval_pids_per_sid = _prepare_mappings(sid_pid2part_seg_label,
                                                                          eval_spec.ignore_label)

    train_sids2train_pids = dict()
    for sid, part_seg_id in zip(sids2part_seg_ids.keys(), sids2part_seg_ids.values()):
        cont_sid = meta["thing_dataset_id_to_contiguous_id"][sid]
        train_sids2train_pids[cont_sid] = part_seg_id

    train_sids_with_parts = np.array(list(train_sids2train_pids.keys()))
    meta["train_sids_with_parts"] = train_sids_with_parts
    meta["train_sids2train_pids"] = train_sids2train_pids

    for key, (image_dir, gt_dir, gt_json) in _RAW_CITYSCAPES_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes_panoptic_parts(x, y, z, meta)
        )

        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir.replace("cityscapes_panoptic_", ""),
            evaluator_type="cityscapes_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )

