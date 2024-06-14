import argparse
import glob
import json
import os
import os.path as op

import numpy as np
from PIL import Image

from panoptic_parts.specs.eval_spec import PartPQEvalSpec
from panoptic_parts.utils.evaluation_PartPQ import evaluate_PartPQ_multicore
from panoptic_parts.utils.utils import get_filenames_in_dir, find_filename_in_list


def filepaths_pairs_fn(basepath_gt, basepath_pred, images_json):
  # returns a list of tuples with paths
  pairs = list()

  with open(images_json, 'r') as fp:
    images_dict = json.load(fp)
  images_list = images_dict['images']

  filenames_gt = get_filenames_in_dir(basepath_gt)
  filenames_pred = get_filenames_in_dir(basepath_pred)

  for image in images_list:
    image_id = image['id']
    image_id = str(image_id)
    fp_gt = find_filename_in_list(image_id, filenames_gt, subject='PPS ground-truth', ext='.tif')
    fp_pred = find_filename_in_list(image_id, filenames_pred, subject='ground-truth PPS', ext='.png')
    pairs.append((fp_gt, fp_pred))

  print(f"Found {len(pairs)} ground truth images and corresponding predictions.")
  return pairs


def pred_reader_fn(fp_pred):
  """
  Read the filepath `fp_pred` and generate the three arrays which conform
  to the specifications of the 3-channel format described in <DESCR-TO-BE-ADDED.md>.
  
  This function assumes that predictions are saved in the 3-channel format
  create your custom function if you use another format.
  """
  part_pred_sample = np.array(Image.open(fp_pred), dtype=np.int32)
  pan_classes = part_pred_sample[..., 0]
  pan_inst_ids = part_pred_sample[..., 1]
  parts_output = part_pred_sample[..., 2]
  return pan_classes, pan_inst_ids, parts_output


def evaluate(eval_spec_path,
             basepath_gt,
             basepath_pred,
             images_json,
             save_dir=None,
             return_results=False,
             cpu_num=None,
             ):
  """

  :param eval_spec_path: filepath of the evaluation specification (EvalSpec)
  :param basepath_gt: directory containing the ground truth files as downloaded from the original source
  :param basepath_pred: directory containing the PPS predictions in the 3-channel PNG format.
  :param images_json: path to the images.json file with a list of images and corresponding image ids.
  :param save_dir: directory where the results should be stored

  :return:
  """
  print('Evaluating the PPS results in {} on the PartPQ metric... (this can take a while)'.format(basepath_pred), flush=True)
  spec = PartPQEvalSpec(eval_spec_path)

  # dst_name = spec._dspec.dataset_name
  filepaths_pairs = filepaths_pairs_fn(basepath_gt, basepath_pred, images_json)

  if cpu_num is None:
    results = evaluate_PartPQ_multicore(spec, filepaths_pairs, pred_reader_fn)
  else:
    results = evaluate_PartPQ_multicore(spec, filepaths_pairs, pred_reader_fn, cpu_num=cpu_num)
  suffix = "_partpq"

  print()
  print(*map(lambda d: ', '.join(map(lambda t: f'{t[0]}: {t[1]:.3f}', d.items())),
             results[0]),
        sep='\n')
  print(*map(lambda t: f'{t[0]:15} ' + ', '.join(map(lambda t: f'{t[0]}: {t[1]:.3f}', t[1].items())),
             zip(spec.eval_sid2scene_label.values(), results[1].values())),
        sep='\n')
  print()

  if save_dir is not None:
    if not op.exists(save_dir):
      print("Creating output directory at {}".format(save_dir), flush=True)
      os.mkdir(save_dir)
    filepath_output = op.join(save_dir, f'{op.basename(eval_spec_path)[:-14]}_results{suffix}.json')
    # if op.exists(filepath_output):
    #   if input(f'Overwrite existing file {filepath_output} (y/n): ') == 'n':
    #     exit()
    with open(filepath_output, 'w') as fp:
      json.dump(results, fp)
    print("Results saved in {}".format(filepath_output), flush=True)

  if return_results:
    return results


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('eval_spec_path', type=str,
                      help="filepath of the evaluation specification, one of 'eval_specs/ppq_*.yaml'")
  parser.add_argument('basepath_gt', type=str,
                      help="directory containing the ground truth files as downloaded from the original source, "
                           "e.g. <prefix_path_on_your_machine> + 'gtFinePanopticParts/val' for CPP or 'validation' for PPP")
  parser.add_argument('basepath_pred', type=str,
                      help="directory containing PNG predictions with filename '<image_id>.png', "
                           "where <image_id> correspond to the filename of the GT files.")
  parser.add_argument('images_json', type=str,
                      help="the json file with a list of images and corresponding image ids.")
  parser.add_argument('--save_dir', type=str,
                      help="directory where the results should be stored", default=None)
  args = parser.parse_args()

  evaluate(args.eval_spec_path, args.basepath_gt, args.basepath_pred, args.images_json, args.save_dir)
