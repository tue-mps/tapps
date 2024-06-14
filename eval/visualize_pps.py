import argparse
import os
import numpy as np
import sys
import psutil

from PIL import Image
from functools import partial

import multiprocessing
from multiprocessing import Pool

sys.path.append("utils/panoptic_parts")
from panoptic_parts.utils.format import encode_ids
from panoptic_parts.utils.visualization import experimental_colorize_label
from panoptic_parts.specs.eval_spec import PartPQEvalSpec


def colorize_pps_and_store_single(file, predictions_path, sid2color, save_path, eval_spec, dataset, cpu_aff=None):
  predictions_np = np.array(Image.open(os.path.join(predictions_path, file)), dtype=np.int32)
  sids = predictions_np[..., 0]
  iids = predictions_np[..., 1]
  pids = predictions_np[..., 2]

  sids_no_parts = eval_spec.eval_sid_no_parts
  sids_stuff = eval_spec.eval_sid_stuff

  sids_wo_parts = np.isin(sids, sids_no_parts)
  sids_w_stuff = np.isin(sids, sids_stuff)
  iids[sids_w_stuff] = -1
  pids[sids_wo_parts] = -1

  sids[sids == 255] = 0
  pids[pids == 255] = -1
  iids[iids == 255] = -1
  uids = encode_ids(sids, iids, pids)

  if cpu_aff is not None:
    process = psutil.Process()
    process.cpu_affinity(cpu_aff)

  if dataset == 'cityscapes' or dataset == 'Cityscapes':
    is_cpp = True
  else:
    is_cpp = False

  pps_colors = experimental_colorize_label(uids, sid2color=sid2color, is_cpp=is_cpp)
  pps_colors_img = Image.fromarray(pps_colors.astype(np.uint8))
  pps_colors_img.save(os.path.join(save_path, file))


def convert_pps_to_colors_and_store(predictions_path, save_path, eval_spec_path, dataset):
  eval_spec = PartPQEvalSpec(eval_spec_path)
  sid2color = eval_spec.dataset_spec.sid2scene_color

  files = list()
  for file in os.listdir(predictions_path):
    if file.endswith(".png"):
      files.append(file)

  if not os.path.exists(save_path):
    os.mkdir(save_path)

  # for file in tqdm(files):
  process = psutil.Process()
  cpu_aff = process.cpu_affinity()

  num_cpus = round(multiprocessing.cpu_count() / 2)

  colorize_pps_and_store_single_fn = partial(colorize_pps_and_store_single,
                                             predictions_path=predictions_path,
                                             sid2color=sid2color,
                                             save_path=save_path,
                                             cpu_aff=cpu_aff,
                                             eval_spec=eval_spec,
                                             dataset=dataset)
  print(f"Now visualizing {len(files)} PPS predictions... this could take a while.")
  with Pool(num_cpus) as p:
    p.map(colorize_pps_and_store_single_fn, files)


def visualize(pred_dir, save_dir, dataset):
  if dataset in ['pascal', 'Pascal', 'pascal57', 'Pascal57']:
    eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml"
  elif dataset in ['pascal107, Pascal107']:
    eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_107_cvpr21_default_evalspec.yaml"
  elif dataset in ['cityscapes', 'Cityscapes']:
    eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml"
  else:
    raise NotImplementedError(f"Only implemented for Pascal, Pascal107 and Cityscapes, not {dataset}.")

  if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

  convert_pps_to_colors_and_store(pred_dir, save_dir, eval_spec_path, dataset=dataset)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--pred_dir', type=str)
  parser.add_argument('--save_dir', type=str)
  parser.add_argument('--dataset', type=str)
  args = parser.parse_args()

  pred_dir = args.pred_dir
  save_dir = args.save_dir
  dataset = args.dataset

  visualize(pred_dir,
            save_dir,
            dataset)
