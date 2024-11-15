import json
import os
from tqdm import tqdm
import shutil
import yaml
import random
from PIL import Image
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
sys.path.append('C:/Users/neals/Desktop/Practicum/utils/coco_utils/')
from coco_dataloader import COCODataset

def convert_coco_format_to_yolo_format(coco_dir, output_dir,image_folder=None, copy_images=False, use_path=True, coco_prefix=''):
    # convert both train and val
    for tar_dir in ['val', 'train']:
        coco_json_file = f'{coco_dir}/{coco_prefix}{tar_dir}_coco.json'
        image_folder = f'{coco_dir}/{tar_dir}/images'

        with open(coco_json_file, 'r') as f:
            coco_json = json.load(f)
        
        os.makedirs(os.path.join(output_dir, tar_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, tar_dir, 'labels'), exist_ok=True)

        for image_entry in tqdm(coco_json['images']):
            image_filename = image_entry['file_name']
            image_width = image_entry['width']
            image_height = image_entry['height']
            image_id = image_entry['id']

            if copy_images:
                if image_folder is None:
                    img_src_path = image_entry['path']
                else:
                    img_src_path = os.path.join(image_folder,image_filename)
                img_dst_path = os.path.join(output_dir,tar_dir,'images', image_filename)
                shutil.copy(img_src_path, img_dst_path)
            
            label_filename = os.path.splitext(image_filename)[0]+'.txt'
            label_path = os.path.join(output_dir, tar_dir, f'{coco_prefix}labels', label_filename)

            # collect all the annotations in the given image
            annotations = [ann for ann in coco_json['annotations'] if ann['image_id'] == image_id]

            with open(label_path, 'w') as label_file:
                # add each annotation to the label file
                for annotation in annotations:
                    category_id = annotation['category_id'] - 1
                    bbox = annotation['bbox']
                    x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

                    # expected format is normalized xc, yc, nw, nh
                    x_center = (x+width/2)/image_width
                    y_center = (y+height/2)/image_height
                    normalized_width = width/image_width
                    normalized_height = height/image_height

                    # make sure valeus are normalized
                    assert x_center <=1 , f'incorrect converion | {x_center, y_center, normalized_width, normalized_height} | {category_id+1}'
                    assert y_center <=1 , f'incorrect converion | {x_center, y_center, normalized_width, normalized_height} | {category_id+1}'
                    assert normalized_width <=1 , f'incorrect converion | {x_center, y_center, normalized_width, normalized_height} | {category_id+1}'
                    assert normalized_height <=1 , f'incorrect converion | {x_center, y_center, normalized_width, normalized_height} | {category_id+1}'

                    label_line = f'{int(category_id)} {x_center:.16f} {y_center:.16f} {normalized_width:.16f} {normalized_height:.16f}\n'
                    label_file.write(label_line)
    
    categories = coco_json['categories']
    class_names = []
    for category in categories:
        class_names.append(category['name'])
    
    yolo_config = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_file_path = f'{output_dir}/{coco_prefix}yolo_config.yaml'

    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yolo_config, yaml_file, default_flow_style=False)

def load_yolo_labels(label_file, img_width, img_height):
    """
    Load YOLO labels from a text file and denormalize them to image size.

    Parameters:
    - label_file: Path to the label file (str)
    - img_width: Width of the image (int)
    - img_height: Height of the image (int)

    Returns:
    - A list of tuples in the form (class_id, 
    - A list of boxes x, y, w, h) with denormalized coordinates.
    """
    #print(f'reading \n {label_file}')
    with open(label_file, 'r') as f:
        lines = [line.strip().split() for line in f]

    #print(f'found {len(lines)} lines')
    #print(lines)
    labels = []
    boxes = []
    for parts in lines:
        class_id = int(parts[0])
        center_x = float(parts[1]) * img_width
        center_y = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height

        # Convert from center_x, center_y, width, height to x, y, w, h
        x = center_x - (width / 2)
        y = center_y - (height / 2)
        boxes.append([x, y, width, height])
        labels.append(class_id)
    #print(labels)
    #print(boxes)
    return labels, boxes

def get_random_yolo_samples(label_folder, image_folder, img_width, img_height, N):
    """
    Pick N random samples from YOLO label files and denormalize them.

    Parameters:
    - label_folder: Folder containing the label files (str)
    - img_width: Width of the images (int)
    - img_height: Height of the images (int)
    - N: Number of random samples to pick (int)

    Returns:
    - A list of N random samples in the form (class_id, x, y, w, h).
    """
    all_files = []
    
    # Loop over all label files in the folder
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            all_files.append(label_file)
    #print(all_files)
    # Pick N random samples
    random_samples = random.sample(all_files, N)
    sample_images = []
    category_ids = []
    annotations =[]
    for sample in random_samples:
        file_path = os.path.join(label_folder, sample)
        labels, boxes = load_yolo_labels(file_path, img_width, img_height)
        image_name = sample.split('.')[0]
        img = Image.open(f'{image_folder}/{image_name}.png').convert('RGB')
        sample_images.append(img)
        category_ids.append(labels)
        annotations.append(boxes)

    return sample_images, category_ids, annotations

def create_yolo_coco_results(model, eval_coco, out_dir, images_dir = None, 
                            transforms = T.Compose([T.ToTensor()]),
                            batch_size =4, shuffle=False, num_workers=4,drop_last=False, device='cpu'):
    model.to(device)
    with open(eval_coco, 'r') as f:
        coco_json = json.load(f)
    images = coco_json['images']

    print('loading dataset and dataloader...')
    coco_dataset = COCODataset(coco_json_path=eval_coco,
                           images_dir=images_dir,
                           transform=transforms)
    dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers,drop_last=drop_last)
    #image, image_id, image_path, bboxes
    annot_id = 1
    annots=[]
    for images, image_ids, image_paths in tqdm(dataloader, desc='batch inf', total=len(dataloader)):
        images.to(device)
        preds = model(images, verbose=False)

        for image, image_id, image_path, result in zip(images, image_ids, image_paths, preds):
            boxes = result.boxes.cpu().numpy()
            if len(boxes) <= 0: continue
            #img_name = image_path.split('/')[-1]
            for box_ind, box in enumerate(boxes):
                cx,cy,width,height = boxes.xywh[box_ind].astype(int)
                x = cx - width/2
                y = cy - height/2
                conf = boxes.conf[box_ind].astype(float)
                category_id = boxes.cls[box_ind].astype(int)

                #print(type(annot_id))
                #print(type(image_id))
                #print(type(category_id))
                #print(type(x),type(y),type(width),type(height))
                #print(type(conf))
                annotation = {
                    "id": int(annot_id), 
                    "image_id": image_id.item(), 
                    "category_id": int(category_id + 1), 
                    #"segmentation": RLE or [polygon],
                    "area": float(width*height), 
                    "bbox": [int(x),int(y),int(width),int(height)], 
                    'score': float(conf),
                    "iscrowd": 0,
                }
                annots.append(annotation)
                annot_id+=1
    
    results_coco = coco_json.copy()
    results_coco['annotations'] = annots

    if not os.path.exists(out_dir): 
        print(f'making dir {out_dir}')
        os.makedirs(out_dir)
        
    with open(f'{out_dir}/YOLO_coco_results_annots.json','w') as f:
        json.dump(results_coco, f, indent=4)
    
    with open(f'{out_dir}/YOLO_coco_results_list.json','w') as f:
        json.dump(annots, f, indent=4)

if __name__ == "__main__":
    '''
    dataset_dir = 'E:/datasets'
    dataset_name = 'DOTA_dataset_512'
    coco_dir = f'{dataset_dir}/{dataset_name}'
    output_dir = coco_dir
    convert_coco_format_to_yolo_format(coco_dir, output_dir,image_folder=None, copy_images=False, use_path=True, coco_prefix='')
    '''