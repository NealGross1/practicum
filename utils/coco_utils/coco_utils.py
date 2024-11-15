import json
import pandas as pd
import os
import sys
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np

def eval_coco(val_coco, pred_coco, iou_thresh=0.45, conf_thresh=0.0):
    # Load COCO ground truth and predictions
    coco_gt = COCO(val_coco)
    coco_dt = coco_gt.loadRes(pred_coco)

    # Filter detections by confidence threshold
    filtered_annots = [dt for dt in coco_dt.dataset['annotations'] if dt['score'] > conf_thresh]
    
    # Update dataset with filtered annotations
    coco_dt.dataset['annotations'] = filtered_annots
    coco_dt.createIndex()  # Reindex the dataset after modification
    
    # Initialize COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    
    # Set IoU threshold
    coco_eval.params.iouThrs = [iou_thresh]
    
    # Evaluate results
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Extract and return detailed results
    results = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    
    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue
        category_id = eval_img['category_id']
        
        # True Positives: correct matches
        tp = (eval_img['dtMatches'][0] > 0).sum()

        # False Positives: detections that didn't match any ground truth
        fp = (eval_img['dtMatches'][0] == 0).sum()

        # False Negatives: ground truth that didn't match any detection
        fn = (eval_img['gtMatches'][0] == 0).sum()

        results[category_id]['TP'] += tp
        results[category_id]['FP'] += fp
        results[category_id]['FN'] += fn
    
    precision_recall = {}
    
    for cat_id, counts in results.items():
        tp = counts['TP']
        fp = counts['FP']
        fn = counts['FN']
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        precision_recall[int(cat_id)] = {'Precision': float(precision), 'Recall': float(recall)}
    
    return precision_recall

class SilentOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_pr_curves_over_conf(val_coco, pred_coco, iou_thresh=0.45, conf_thresholds=np.arange(0.05, 1.0, 0.05)):
    pr_curves = defaultdict(lambda: {'precision': [], 'recall': [], 'conf_thresh': []})
    
    for conf_thresh in tqdm(conf_thresholds):
        # Get precision and recall at the current confidence threshold
        with SilentOutput():
            precision_recall = eval_coco(val_coco, pred_coco, iou_thresh=iou_thresh, conf_thresh=conf_thresh)
        
        for cat_id, pr in precision_recall.items():
            pr_curves[cat_id]['precision'].append(pr['Precision'])
            pr_curves[cat_id]['recall'].append(pr['Recall'])
            pr_curves[cat_id]['conf_thresh'].append(conf_thresh)
    
    return pr_curves
    

def create_df_from_coco(coco_json_file):
    with open(coco_json_file, 'r') as f:
        coco_json = json.load(f)
    
    image_id_to_image_name_map = {image_dict['id']:image_dict['file_name'] for image_dict in coco_json['images']}
    image_id_to_path_map = {image_dict['id']:image_dict['path'] for image_dict in coco_json['images']}
    category_id_to_name = {category_dict['id']:category_dict['name'] for category_dict in coco_json['categories']}
    df_dict_list = []
    for annot_dict in coco_json['annotations']:
        file_name = image_id_to_image_name_map[annot_dict['image_id']]
        path = image_id_to_path_map[annot_dict['image_id']]
        coco_bbox = annot_dict['bbox']
        xy_bbox = [coco_bbox[0],coco_bbox[1],coco_bbox[0]+coco_bbox[2],coco_bbox[1]+coco_bbox[3]]
        df_dict_list.append({
            'image_file_name':file_name,
            'id': annot_dict['id'],
            'image_id':annot_dict['image_id'],
            'image_file_path':path,
            'category_id': int(annot_dict['category_id']),
            'class_name': category_id_to_name[annot_dict['category_id']],
            'xy_bbox': xy_bbox,
            'coco_bbox': coco_bbox,
            'segmentation': annot_dict['segmentation']
        })
    
    return pd.DataFrame(df_dict_list)

def add_path_to_coco(coco_path, image_dir):
    with open(coco_path, 'r') as f:
        coco_json = json.load(f)

    new_image_dicts = []

    for image_dict in coco_json['images']:
        assert os.path.exists(f'{image_dir}/{image_dict['file_name']}'), f'image not found {image_dir}/{image_dict['file_name']}'
        image_dict['path'] = f'{image_dir}/{image_dict['file_name']}'
        new_image_dicts.append(image_dict)
    
    coco_json['images'] = new_image_dicts

    with open(coco_path, 'w') as f:
        json.dump(coco_json, f, indent=4)

    
if __name__ == "__main__":
    dataset_dir = 'E:/datasets'
    dataset_name = 'DOTA_dataset'
    coco_dir = f'{dataset_dir}/{dataset_name}'
    add_path_to_coco(f'{coco_dir}/train_coco.json', f'{coco_dir}/train/images')
    add_path_to_coco(f'{coco_dir}/val_coco.json',  f'{coco_dir}/val/images')