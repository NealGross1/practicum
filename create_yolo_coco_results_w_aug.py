from ultralytics import YOLO
from utils.yolo_utils.yolo_utils import create_yolo_coco_results
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
from PIL import Image
import numpy as np
import random



class DropPixels(A.ImageOnlyTransform):
    def __init__(self, drop_fraction=0.01, always_apply=False, p=1.0):
        super(DropPixels, self).__init__(always_apply, p)
        self.drop_fraction = drop_fraction

    def apply(self, img, **params):
        return self.drop_pixels(img, self.drop_fraction)

    def drop_pixels(self, image, drop_fraction):
        height, width, _ = image.shape
        num_pixels_to_drop = int(width * height * drop_fraction)

        for _ in range(num_pixels_to_drop):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            image[y, x] = [0, 0, 0]  # Drop to black (or change to another color if needed)
        
        return image
    
class AlbumentationsTransform:
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, image):
            image = np.array(image)  # Convert PIL image to NumPy array
            augmented = self.transform(image=image)
            return Image.fromarray(augmented['image'])  # Convert NumPy array back to PIL image

if __name__ == "__main__":
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        device = 'cuda'
    else:
        device = 'cpu'

    dataset_dir = 'E:/datasets'
    dataset_name = 'DOTA_dataset_512'
    coco_dir = f'{dataset_dir}/{dataset_name}'
    eval_coco = f'{coco_dir}/val_coco.json'
    output_dir = coco_dir
    model_dir = 'E:/models'
    model_name = 'DOTA_dataset_512_yolov8m_bb_aug'
    model_path = f'{model_dir}/{model_name}/weights/best.pt'
    model = YOLO(model_path)
    batch_size = 64
    num_workers = 8

    print(f'starting creation of coco results for {eval_coco} with model {model_path} using {device}')
    # Custom wrapper for Albumentations
    

    print('blur results')
    output_dir_specific = f'{output_dir}/blur'
    # Example Albumentations transformation
    albumentations_transform = A.Compose([
      A.GaussianBlur(blur_limit=(7, 7), p=1.0)
    ])

    # Combine Albumentations and Torchvision transforms
    transform = T.Compose([
        AlbumentationsTransform(albumentations_transform),
        T.ToTensor()  # This works with PIL images
    ])

    create_yolo_coco_results(model, 
                                eval_coco, 
                                output_dir_specific, 
                                images_dir = None, 
                                transforms = transform,
                                batch_size = batch_size, 
                                shuffle=False, 
                                num_workers=num_workers,
                                drop_last=False, 
                                device=device)
    
    print('rotation results')
    output_dir_specific = f'{output_dir}/rotation'
     # Example Albumentations transformation
    albumentations_transform = A.Compose([
      A.Rotate(limit=(45, 45), p=1.0)
    ])
    # Combine Albumentations and Torchvision transforms
    transform = T.Compose([
        AlbumentationsTransform(albumentations_transform),
        T.ToTensor()  # This works with PIL images
    ])

    create_yolo_coco_results(model, 
                                eval_coco, 
                                output_dir_specific, 
                                images_dir = None, 
                                transforms = transform,
                                batch_size = batch_size, 
                                shuffle=False, 
                                num_workers=num_workers,
                                drop_last=False, 
                                device=device)
    
    print('drop pixel results')
    output_dir_specific = f'{output_dir}/drop_pixel'
     # Example Albumentations transformation
    albumentations_transform = A.Compose([
      DropPixels(drop_fraction=0.01, p=1.0),
    ])
    # Combine Albumentations and Torchvision transforms
    transform = T.Compose([
        AlbumentationsTransform(albumentations_transform),
        T.ToTensor()  # This works with PIL images
    ])

    create_yolo_coco_results(model, 
                                eval_coco, 
                                output_dir_specific, 
                                images_dir = None, 
                                transforms = transform,
                                batch_size = batch_size, 
                                shuffle=False, 
                                num_workers=num_workers,
                                drop_last=False, 
                                device=device)