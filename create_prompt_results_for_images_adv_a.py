from transformers import pipeline,  AutoProcessor, AutoModelForSeq2SeqLM, \
    BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration 
import torch
from PIL import Image
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import albumentations as A
import random
import numpy as np
import torchvision.transforms as transforms

def save_to_csv(df_list, csv_path):
    df = pd.DataFrame(df_list)
    df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)

# Define augmentation functions
def rotate_image(image, angle):
    transform = A.Rotate(limit=(angle, angle), p=1.0)
    return transform(image=image)['image']

def gaussian_blur_image(image, sigma):
    transform = A.GaussianBlur(blur_limit=(sigma, sigma), p=1.0)
    return transform(image=image)['image']

def drop_pixels(image, drop_fraction=0.01):
    height, width, _ = image.shape
    num_pixels_to_drop = int(width * height * drop_fraction)
    
    for _ in range(num_pixels_to_drop):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        image[y, x] = [0, 0, 0]  # Drop to black or any other color
    
    return image


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to(device) 

    dataset_dir = 'E:/datasets'
    dataset_name = 'DOTA_dataset_512'
    coco_dir = f'{dataset_dir}/{dataset_name}'
    coco_file = f'{coco_dir}/YOLO_coco_results_annots.json'
    prompts = ["A picture of", "A photo of", "The image quality is"]

    with open(coco_file, 'r') as f:
        coco_json = json.load(f)
    
    image_id_to_annots = defaultdict(list)
    #cats_to_use = [1,2,3,4,12,15]
    for annot in coco_json['annotations']:
    #    if annot['category_id'] in cats_to_use:
            image_id_to_annots[annot['image_id']].append(annot)

    init_data = defaultdict(int)
    for category in coco_json['categories']:
        init_data[category['id']] = 0

    df_list = []
    csv_path = 'prompt_results_for_chips_adv_a.csv'
    
    for image_dict in tqdm(coco_json['images'], desc='Processing images', leave=False):
        image_path = image_dict['path']
        
        image_annotations = image_id_to_annots.get(image_dict['id'], [])
        
        # Only proceed if there are annotations for this image
        if len(image_annotations) > 0:
            category_counts = init_data.copy()
            for annot in image_annotations:
                category_counts[annot['category_id']] += 1

            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)

            # Process the image with each prompt
            for prompt in prompts:

                rotated_image = Image.fromarray(rotate_image(img_np, 45))
                #rotated_image=rotated_image.to(device)
                inputs = processor(rotated_image, text=prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                data = {
                    'image_id': image_dict['id'],
                    'image_file': image_dict['file_name'],
                    'image_path': image_dict['path'],
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'augmentation': 'rotation'
                }
                
                data.update(category_counts)
                df_list.append(data)

                blur_image = Image.fromarray(gaussian_blur_image(img_np, 7))
                #blur_image=blur_image.to(device)
                inputs = processor(blur_image, text=prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                data = {
                    'image_id': image_dict['id'],
                    'image_file': image_dict['file_name'],
                    'image_path': image_dict['path'],
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'augmentation': 'blur'
                }
                
                data.update(category_counts)
                df_list.append(data)

                pixel_drop_img = Image.fromarray(drop_pixels(img_np, 0.001))
                #pixel_drop_img=pixel_drop_img.to(device)
                inputs = processor(pixel_drop_img, text=prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                data = {
                    'image_id': image_dict['id'],
                    'image_file': image_dict['file_name'],
                    'image_path': image_dict['path'],
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'augmentation': 'drop_pixel'
                }
                
                data.update(category_counts)
                df_list.append(data)
        
        # Save periodically (e.g., after processing each image)
        save_to_csv(df_list, csv_path)
        df_list = []  # Clear the list to free up memory

    # Save any remaining data in the list
    if df_list:
        save_to_csv(df_list, csv_path)

