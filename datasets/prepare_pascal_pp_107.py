import os
import sys
import psutil
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

eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_107_cvpr21_default_evalspec.yaml"
eval_spec = PartPQEvalSpec(eval_spec_path)
IGNORE_VALUE = 255

root = os.getenv("DETECTRON2_DATASETS", "datasets")
gt_dir_pps = os.path.join(root, "pascal/labels/")
gt_dir_parts = os.path.join(root, "pascal/parts_107/")
splits = ["validation", "training"]

if not os.path.isdir(gt_dir_parts):
  os.mkdir(gt_dir_parts)
else:
  raise FileExistsError(f"The directory {gt_dir_parts} already exists. Running this command would overwrite this directory. Delete first before running again.")


eval_sid_things = eval_spec.eval_sid_things
eval_sid_stuff = eval_spec.eval_sid_stuff
eval_sid_total = list(set(eval_sid_stuff + eval_sid_things))

categories = list()
for eval_sid in eval_sid_total:
  isthing = 1 if eval_sid in eval_sid_things else 0
  categories.append({"id": eval_sid,
                     "isthing": isthing,
                     "color": eval_spec.dataset_spec.scene_color_from_sid(eval_sid),
                     "name": eval_spec.eval_sid2scene_label[eval_sid]})

for split in splits:
  gt_dir_pps_split = os.path.join(gt_dir_pps, split)
  gt_dir_parts_split = os.path.join(gt_dir_parts, split)
  if not os.path.isdir(gt_dir_parts_split):
    os.mkdir(gt_dir_parts_split)

  file_names = list()
  for file in os.listdir(gt_dir_pps_split):
      if file.endswith(".tif"):
          file_names.append(file)

  def retrieve_ignore_info_parts(sid, sid_iid, sid_pid, eval_spec):
    # Retrieve ignore mask for part segmentation
    ignore_img = np.zeros_like(sid).astype(np.bool_)

    # ignore parts that are not part of an instance (should always be the case for Pascal PP)
    no_iid = sid_iid < 1000
    with_parts = np.isin(sid, eval_spec.eval_sid_parts)
    things = np.isin(sid, eval_spec.eval_sid_things)
    things_and_parts = np.logical_and(with_parts, things)
    crowd = np.logical_and(no_iid, things_and_parts)
    ignore_img[crowd] = True

    # ignore regions without part labels
    no_pid = sid_pid < 100
    ignore_parts = np.logical_and(no_pid, with_parts)
    ignore_img[ignore_parts] = True

    return ignore_img


  def convert_and_store(pps_filename, cpu_affinity):
    process = psutil.Process()
    process.cpu_affinity(cpu_affinity)

    part_seg_gt_uids = np.array(Image.open(os.path.join(gt_dir_pps_split, pps_filename)))

    sids, iids, pids, sids_iids, sids_pids = pp.decode_uids(part_seg_gt_uids,
                                                            return_sids_iids=True,
                                                            return_sids_pids=True,
                                                            experimental_dataset_spec=eval_spec._dspec)

    dataset_sid_pid2eval_sid_pid = parse_dataset_sid_pid2eval_sid_pid(eval_spec.dataset_sid_pid2eval_sid_pid)
    dataset_sid_pid2eval_sid_pid = ndarray_from_dict(dataset_sid_pid2eval_sid_pid, -10 ** 6,
                                                     length=10000)  # -10**6: a random big number

    eval_sids_pids = dataset_sid_pid2eval_sid_pid[sids_pids]

    ignore_region_parts = retrieve_ignore_info_parts(sids, sids_iids, eval_sids_pids, eval_spec)

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
    file_name_save = os.path.join(gt_dir_parts_split, pps_filename)
    file_name_save = file_name_save.replace('.tif', '.png')
    img_part_seg_gt.save(file_name_save)

  process = psutil.Process()
  cpu_aff = process.cpu_affinity()
  convert_and_store_fn = partial(convert_and_store, cpu_affinity=cpu_aff)

  print(f"Creating part labels for {split} split. This may take a while...")
  pool = Pool()  # Create a multiprocessing Pool
  pool.map(convert_and_store_fn, file_names)

print("Finished.")
