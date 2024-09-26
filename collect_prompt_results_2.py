import torch
from PIL import Image
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
#https://github.com/salesforce/LAVIS
from lavis.models import load_model_and_preprocess
from utils.coco_utils.annotate_images import annotate_img
from time import time
import os 



def save_to_csv(df_list, csv_path):
    df = pd.DataFrame(df_list)
    df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('loading model')
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

    dataset_dir = 'E:/datasets'
    dataset_name = 'DOTA_dataset_512'
    coco_dir = f'{dataset_dir}/{dataset_name}'
    coco_file = f'{coco_dir}/val_coco.json'

    objects_of_interest = ['car','truck','ship','plane','storage-tank',"swimming-pool","harbor"]
    object_count_prompts = [f"Question: How many {obj}s are there? Answer:" for obj in objects_of_interest]
    object_classification_prompts = [f"Question: Is there a {obj}? Answer:" for obj in objects_of_interest]
    object_classification_ann_prompts = [f"Question: Is there a {obj} in the red box? Answer:" for obj in objects_of_interest]

    prompts = ["","Question: is there a plane in this image, If you are not sure about the answer, say you don't know? Answer:",
               "Question: is anything moving? Answer:"]

    with open(coco_file, 'r') as f:
        coco_json = json.load(f)

    image_id_to_annots = defaultdict(list)
    cats_to_use = [1,2,3,4,12,15]
    for annot in coco_json['annotations']:
        if annot['category_id'] in cats_to_use:
            image_id_to_annots[annot['image_id']].append(annot)

    image_id_to_dict_map ={}
    for image_dict in coco_json['images']:
        image_id_to_dict_map[image_dict['id']] = image_dict
    
    
    csv_path = 'prompt_results_classification.csv'

    if os.path.exists(csv_path): 
        print('WARNING existing data need to configure to modify or delete')
        exit

    for img_id, image_dict in tqdm(image_id_to_dict_map.items(),desc='Img',leave=False,total=len(image_id_to_dict_map.keys())):
        #saving each loop
        df_list = []
        if len(image_id_to_annots[img_id]) <= 0 : continue
        image_path = image_dict['path']
        img = Image.open(image_path).convert('RGB')
        image = vis_processors["eval"](img).unsqueeze(0).to(device)
        for prompt in tqdm(prompts, desc='p1',leave=False):
            start_time = time()
            output = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=False)[0]
            end_time = time()
            df_list.append({
                        'image_id': img_id,
                        'image_file': image_dict['file_name'],
                        'image_path': image_path,
                        'prompt': prompt,
                        'generated_text': output,
                        'category':annot['category_id'],
                        'processing_time': end_time - start_time
                    })
        
        for classification_p, count_p in tqdm(zip(object_classification_prompts,object_count_prompts), desc='p2',leave=False):
            start_time = time()
            output = model.generate({"image": image, "prompt": classification_p}, use_nucleus_sampling=False)[0]
            end_time = time()
            df_list.append({
                        'image_id': img_id,
                        'image_file': image_dict['file_name'],
                        'image_path': image_path,
                        'prompt': classification_p,
                        'generated_text': output,
                        'category':annot['category_id'],
                        'processing_time': end_time - start_time
                    })
            if output != 'No':
                start_time = time()
                output = model.generate({"image": image, "prompt": count_p}, use_nucleus_sampling=False)[0]
                end_time = time()
                df_list.append({
                            'image_id': img_id,
                            'image_file': image_dict['file_name'],
                            'image_path': image_path,
                            'prompt': count_p,
                            'generated_text': output,
                            'category':annot['category_id'],
                            'processing_time': end_time - start_time
                        })
        # Save periodically (e.g., after processing each image)
        save_to_csv(df_list, csv_path)


    # Save any remaining data in the list
    if df_list:
        save_to_csv(df_list, csv_path)


