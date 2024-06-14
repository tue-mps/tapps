import os
import shutil
import psutil
import sys
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from functools import partial

from multiprocessing import Pool

sys.path.append("utils/panoptic_parts")
import panoptic_parts as pp
from panoptic_parts.specs.eval_spec import PartPQEvalSpec
from panoptic_parts.utils.utils import _sparse_ids_mapping_to_dense_ids_mapping
from panoptic_parts.utils.evaluation_PartPQ import parse_dataset_sid_pid2eval_sid_pid
from panoptic_parts.utils.utils import _sparse_ids_mapping_to_dense_ids_mapping as ndarray_from_dict

eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml"
eval_spec = PartPQEvalSpec(eval_spec_path)
IGNORE_VALUE = 255

root = os.getenv("DETECTRON2_DATASETS", "datasets")
gt_dir_pps = os.path.join(root, "pascal/labels/")
gt_dir_panoptic = os.path.join(root, "pascal/panoptic/")
gt_dir_semantic = os.path.join(root, "pascal/semantic/")
gt_dir_parts = os.path.join(root, "pascal/parts/")
gt_dir_images_json = os.path.join(root, "pascal/")
img_dir_orig = os.path.join(root, "pascal/JPEGImages/")
img_dir_new = os.path.join(root, "pascal/images/")
splits = ["validation", "training"]

for dir_path in [gt_dir_panoptic, gt_dir_semantic, gt_dir_parts, img_dir_new]:
  if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
  else:
    raise FileExistsError(f"The directory {dir_path} already exists. Running this command would overwrite this directory. Delete first before running again.")

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
  print(f"Creating data splits and generating panoptic and semantic labels for {split} split...")
  gt_dir_pps_split = os.path.join(gt_dir_pps, split)
  gt_dir_panoptic_split = os.path.join(gt_dir_panoptic, split)
  gt_dir_semantic_split = os.path.join(gt_dir_semantic, split)
  gt_dir_parts_split = os.path.join(gt_dir_parts, split)
  img_dir_new_split = os.path.join(img_dir_new, split)
  if not os.path.isdir(gt_dir_panoptic_split):
    os.mkdir(gt_dir_panoptic_split)
  if not os.path.isdir(gt_dir_semantic_split):
    os.mkdir(gt_dir_semantic_split)
  if not os.path.isdir(gt_dir_parts_split):
    os.mkdir(gt_dir_parts_split)
  if not os.path.isdir(img_dir_new_split):
    os.mkdir(img_dir_new_split)

  file_names = list()
  for file in os.listdir(gt_dir_pps_split):
      if file.endswith(".tif"):
          file_names.append(file)

  annotations = list()
  images = list()

  for file_name in tqdm(file_names):
    # Copy image files to correct folder
    file_name_image = file_name.replace(".tif", ".jpg")
    shutil.copyfile(src=os.path.join(img_dir_orig, file_name_image), dst=os.path.join(img_dir_new_split, file_name_image))

    file_name_anno = file_name.replace(".tif", ".png")
    gt_pps_path = os.path.join(gt_dir_pps_split, file_name)
    gt_pps_uids = np.array(Image.open(gt_pps_path))
    sids, iids, _, sids_iids = pp.decode_uids(gt_pps_uids, return_sids_iids=True)
    unique_sids_iids = np.unique(sids_iids)

    panoptic_img = np.zeros((sids.shape[0], sids.shape[1], 3), dtype=np.int32)
    semantic_img = np.ones((sids.shape[0], sids.shape[1]), np.uint8) * 255

    segments_info = list()
    for sid_iid in unique_sids_iids:
      segment_id = sid_iid
      if segment_id < 1000:
        if segment_id not in eval_sid_total:
          continue
        if segment_id in eval_sid_things:
          # segment is thing, but crowd
          iscrowd = 1
          category_id = segment_id
        else:
          iscrowd = 0
          category_id = segment_id

      else:
        sid = segment_id // 1000
        if sid not in eval_sid_total:
          continue
        if sid not in eval_sid_things:
          raise ValueError("sid_iid is {} so has an instance id, but sid {} is not a things class.".format(sid_iid, sid))
        else:
          iscrowd = 0
          category_id = segment_id // 1000

      mask = sids_iids == segment_id
      color = [segment_id % 256, segment_id // 256, segment_id // 256 // 256]
      panoptic_img[mask] = color

      semantic_img[mask] = category_id

      area = np.sum(mask)  # segment area computation

      # bbox computation for a segment
      hor = np.sum(mask, axis=0)
      hor_idx = np.nonzero(hor)[0]
      x = hor_idx[0]
      width = hor_idx[-1] - x + 1
      vert = np.sum(mask, axis=1)
      vert_idx = np.nonzero(vert)[0]
      y = vert_idx[0]
      height = vert_idx[-1] - y + 1
      bbox = [int(x), int(y), int(width), int(height)]

      segments_info.append({"id": int(segment_id),
                            "category_id": int(category_id),
                            "area": int(area),
                            "bbox": bbox,
                            "iscrowd": iscrowd})

    h, w = panoptic_img.shape[0:2]
    image_id = file_name.replace(".tif", "")
    annotations.append({"file_name": file_name_anno,
                        "image_id": image_id,
                        "segments_info": segments_info,
                         })

    images.append({"file_name": file_name_anno,
                   "id": image_id,
                   "height": int(h),
                   "width": int(w)})

    gt_pan_path = os.path.join(gt_dir_panoptic_split, image_id + '.png')
    Image.fromarray(panoptic_img.astype(np.uint8)).save(gt_pan_path)
    gt_sem_path = os.path.join(gt_dir_semantic_split, image_id + '.png')
    Image.fromarray(semantic_img.astype(np.uint8)).save(gt_sem_path)

  panoptic_dict = {'annotations': annotations,
                   'images': images,
                   'categories': categories}

  images_dict = {'images': images}

  # Store json
  gt_pan_json_path = os.path.join(gt_dir_panoptic, 'panoptic_' + split + '.json')
  with open(gt_pan_json_path, 'w') as fp:
    json.dump(panoptic_dict, fp)

  images_json_path = os.path.join(gt_dir_images_json, f"images_{split}.json")
  with open(images_json_path, 'w') as fp:
    json.dump(images_dict, fp)

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
