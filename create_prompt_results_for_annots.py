from transformers import pipeline,  AutoProcessor, AutoModelForSeq2SeqLM, \
    BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration 
import torch
from PIL import Image
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def extract_region_with_buffer(image, xyxy_box, buffer=0):
    # Unpack the xyxy box
    x_min, y_min, x_max, y_max = xyxy_box
    
    # Add buffer
    x_min = max(x_min - buffer, 0)
    y_min = max(y_min - buffer, 0)
    x_max = min(x_max + buffer, image.width)
    y_max = min(y_max + buffer, image.height)
    
    # Crop the image
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    
    return cropped_image

def save_to_csv(df_list, csv_path):
    df = pd.DataFrame(df_list)
    df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)


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
    prompts = ["A picture of"]

    with open(coco_file, 'r') as f:
        coco_json = json.load(f)
    
    image_id_to_annots = defaultdict(list)
    cats_to_use = [1,2,3,4,12,15]
    for annot in coco_json['annotations']:
        if annot['category_id'] in cats_to_use:
            image_id_to_annots[annot['image_id']].append(annot)
    
    df_list = []
    csv_path = 'prompt_results.csv'
    
    for image_dict in tqdm(coco_json['images'], desc='Processing images', leave=False):
        image_path = image_dict['path']
        img = Image.open(image_path).convert('RGB')
        image_annotations = image_id_to_annots.get(image_dict['id'], [])
        if len(image_annotations) > 0:
            for prompt in prompts:
                for annot in tqdm(image_annotations, desc='extracting regions', leave=False):
                    x, y, w, h = annot['bbox']
                    annot_xyxy = [x, y, x+w, y+h]
                    cropped_img = extract_region_with_buffer(img, annot_xyxy, buffer=3)
                    inputs = processor(cropped_img, text=prompt, return_tensors="pt").to(device, torch.float16)
                    #max_new_tokens=40
                    generated_ids = model.generate(**inputs)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    df_list.append({
                        'image_id': image_dict['id'],
                        'image_file': image_dict['file_name'],
                        'image_path': image_dict['path'],
                        'bbox': annot['bbox'],
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'category':annot['category_id'],
                        'score': annot['score']
                    })
        
        # Save periodically (e.g., after processing each image)
        save_to_csv(df_list, csv_path)
        df_list = []  # Clear the list to free up memory

    # Save any remaining data in the list
    if df_list:
        save_to_csv(df_list, csv_path)

