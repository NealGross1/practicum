import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torchvision.transforms as T

class COCODataset(Dataset):
    def __init__(self, coco_json_path, images_dir=None, transform=None):
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.transform = transform

        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Create an index to map image_ids to image file names and annotations
        self.image_ids = []
        self.image_info = {}
        self.image_annotations = {}

        for img in self.coco_data['images']:
            image_id = img['id']
            self.image_ids.append(image_id)
            self.image_info[image_id] = img

        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann['bbox'])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.images_dir is None:
            image_path = self.image_info[image_id]['path']
        else:
            image_path = f"{self.images_dir}/{self.image_info[image_id]['file_name']}"
        image = Image.open(image_path).convert("RGB")
        # Get bounding boxes (each bbox is [x, y, width, height])
        #bboxes = self.image_annotations.get(image_id, [])
        if self.transform:
            image = self.transform(image)

            
        return image, image_id, image_path#, bboxes