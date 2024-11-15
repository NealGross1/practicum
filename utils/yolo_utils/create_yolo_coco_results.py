from ultralytics import YOLO
from yolo_utils import create_yolo_coco_results
import torch
import torch


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
    #eval_coco = f'{coco_dir}/plane_ship_aug_val_coco.json'
    output_dir = coco_dir
    #output_dir = f'{coco_dir}/plane_ship_aug_results'
    model_dir = 'E:/models'
    model_name = 'DOTA_dataset_512_yolov8m_bb_aug'
    model_path = f'{model_dir}/{model_name}/weights/best.pt'
    model = YOLO(model_path)
    batch_size = 64

    print(f'starting creation of coco results for {eval_coco} with model {model_path} using {device}')

    create_yolo_coco_results(model, 
                                eval_coco, 
                                output_dir, 
                                images_dir = None, 
                                #transforms = torch.Compose([torch.ToTensor()]),
                                batch_size = batch_size, 
                                shuffle=False, 
                                num_workers=4,
                                drop_last=False, 
                                device=device)