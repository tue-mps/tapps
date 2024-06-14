import sys
import os
import argparse

sys.path.append("utils/panoptic_parts")
from panoptic_parts.evaluation import eval_PartPQ

def eval_partpq(save_dir, dataset):
  root = os.getenv("DETECTRON2_DATASETS", "datasets")

  if dataset in ['pascal', 'Pascal', 'pascal57', 'Pascal57']:
    eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml"
    gt_path = os.path.join(root, "pascal/labels/validation")
    images_json = os.path.join(root, "pascal/images_validation.json")
  elif dataset in ['pascal107, Pascal107']:
    eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_ppp_59_107_cvpr21_default_evalspec.yaml"
    gt_path = os.path.join(root, "pascal/labels/validation")
    images_json = os.path.join(root, "pascal/images_validation.json")
  elif dataset in ['cityscapes', 'Cityscapes']:
    eval_spec_path = "utils/panoptic_parts/panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml"
    gt_path = os.path.join(root, "cityscapes/gtFinePanopticParts/val/")
    images_json = os.path.join(root, "cityscapes/images_val.json")
  else:
    raise NotImplementedError(f"Only implemented for Pascal, Pascal107 and Cityscapes, not {dataset}.")

  pps_pred_path = os.path.join(save_dir, "pps")
  results_dir = os.path.join(save_dir, "results")

  # Eval PPS predictions with PartPQ
  results = eval_PartPQ.evaluate(eval_spec_path,
                                 gt_path,
                                 pps_pred_path,
                                 images_json,
                                 save_dir=results_dir,
                                 return_results=True)

  part_pq = results[0][0]["PartPQ"]
  part_pq_p = results[0][1]["PartPQ_parts"]
  part_pq_np = results[0][2]["PartPQ_noparts"]
  metrics = {"part_pq": part_pq, "part_pq_p": part_pq_p, "part_pq_np": part_pq_np}
  print(metrics)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str)
  parser.add_argument('--dataset', type=str)
  args = parser.parse_args()

  save_dir = args.save_dir
  dataset = args.dataset

  eval_partpq(save_dir,
              dataset)
