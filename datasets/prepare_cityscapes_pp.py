import sys
import psutil
import os
import json
import numpy as np
from functools import partial
from PIL import Image

from multiprocessing import Pool

sys.path.append("utils/panoptic_parts")
import panoptic_parts as pp
from panoptic_parts.specs.eval_spec import PartPQEvalSpec
from panoptic_parts.utils.utils import _sparse_ids_mapping_to_dense_ids_mapping
from panoptic_parts.utils.evaluation_PartPQ import parse_dataset_sid_pid2eval_sid_pid
from panoptic_parts.utils.utils import _sparse_ids_mapping_to_dense_ids_mapping as ndarray_from_dict

from detectron2.utils.file_io import PathManager

eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml"
eval_spec = PartPQEvalSpec(eval_spec_path)
IGNORE_VALUE = 255

root = os.getenv("DETECTRON2_DATASETS", "datasets")
gt_dir_pps = os.path.join(root, "cityscapes/gtFinePanopticParts/")
gt_dir_parts = os.path.join(root, "cityscapes/gtFineParts/")
gt_dir_images_json = os.path.join(root, "cityscapes/")
splits = ["train", "val"]
if not os.path.isdir(gt_dir_parts):
  os.mkdir(gt_dir_parts)
else:
  raise FileExistsError(f"The directory {gt_dir_parts} already exists. Running this command would overwrite this directory. Delete first before running again.")

for split in splits:
  print(f"Creating part labels for {split} split. This may take a while...")
  gt_dir_pps_split = os.path.join(gt_dir_pps, split)
  gt_dir_parts_split = os.path.join(gt_dir_parts, split)
  if not os.path.exists(gt_dir_parts_split):
    os.mkdir(gt_dir_parts_split)

  images_list = list()

  cities = PathManager.ls(gt_dir_pps_split)

  cities_dir = list()
  for city in cities:
    if os.path.isdir(os.path.join(gt_dir_pps_split, city)):
      cities_dir.append(city)

  for city in cities_dir:
    gt_dir_pps_split_city = os.path.join(gt_dir_pps_split, city)

    file_names = list()
    for file in os.listdir(gt_dir_pps_split_city):
      if file.endswith(".tif"):
        file_names.append(file)

    gt_dir_parts_split_city = os.path.join(gt_dir_parts_split, city)
    if not os.path.exists(gt_dir_parts_split_city):
      os.mkdir(gt_dir_parts_split_city)

    def retrieve_ignore_info_parts(sid, sid_iid, sid_pid, eval_spec):
      # Retrieve ignore mask for part segmentation
      ignore_img = np.zeros_like(sid).astype(np.bool_)

      # ignore crowd region from parts classes
      no_iid = sid_iid < 1000
      with_parts = np.isin(sid, eval_spec.eval_sid_parts)
      things = np.isin(sid, eval_spec.eval_sid_things)
      things_and_parts = np.logical_and(things, with_parts)
      crowd = np.logical_and(no_iid, things_and_parts)
      ignore_img[crowd] = True

      # if pid == 0 and sid in l_parts, set to ignore
      no_pid = sid_pid < 100
      ignore_parts = np.logical_and(no_pid, with_parts)
      ignore_img[ignore_parts] = True

      return ignore_img


    def convert_and_store(pps_filename, cpu_affinity):
      process = psutil.Process()
      process.cpu_affinity(cpu_affinity)

      part_seg_gt_uids = np.array(Image.open(os.path.join(gt_dir_pps_split_city, pps_filename)))

      sids, iids, pids, sids_iids, sids_pids = pp.decode_uids(part_seg_gt_uids,
                                                              return_sids_iids=True,
                                                              return_sids_pids=True,
                                                              experimental_dataset_spec=eval_spec._dspec)

      ignore_region_parts = retrieve_ignore_info_parts(sids, sids_iids, sids_pids, eval_spec)

      dataset_sid_pid2eval_sid_pid = parse_dataset_sid_pid2eval_sid_pid(eval_spec.dataset_sid_pid2eval_sid_pid)
      dataset_sid_pid2eval_sid_pid = ndarray_from_dict(dataset_sid_pid2eval_sid_pid, -10 ** 6,
                                                       length=10000)  # -10**6: a random big number

      eval_sids_pids = dataset_sid_pid2eval_sid_pid[sids_pids]

      eval_sid_pid2part_ids = eval_spec.eval_sid_pid2eval_pid_flat
      eval_sid_pid2part_ids = _sparse_ids_mapping_to_dense_ids_mapping(eval_sid_pid2part_ids,
                                                                       void=IGNORE_VALUE)

      part_seg_gt = eval_sid_pid2part_ids[eval_sids_pids]
      part_seg_gt = part_seg_gt.astype(np.uint8)

      # Replace 255 with background value 0
      part_seg_gt[part_seg_gt == IGNORE_VALUE] = 0

      # Add ignore info
      part_seg_gt[ignore_region_parts] = IGNORE_VALUE

      img_part_seg_gt = Image.fromarray(part_seg_gt)
      file_name_save = os.path.join(gt_dir_parts_split_city, pps_filename)
      file_name_save = file_name_save.replace('.tif', '.png')
      img_part_seg_gt.save(file_name_save)

    process = psutil.Process()
    cpu_aff = process.cpu_affinity()
    convert_and_store_fn = partial(convert_and_store, cpu_affinity=cpu_aff)

    pool = Pool()  # Create a multiprocessing Pool
    pool.map(convert_and_store_fn, file_names)

    for file_name in file_names:
      image_dict = {"file_name": file_name.replace("_gtFinePanopticParts.tif", "_gtFine_panoptic.png"),
                    "id": file_name.replace("_gtFinePanopticParts.tif", ""),
                    "width": 2048,
                    "height": 1024}
      images_list.append(image_dict)

  image_dict = {"images": images_list}
  with open(os.path.join(gt_dir_images_json, f"images_{split}.json"), "w") as f:
    json.dump(image_dict, f)


print("Finished.")
