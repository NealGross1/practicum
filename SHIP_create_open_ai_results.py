#https://platform.openai.com/docs/quickstart?desktop-os=windows&language-preference=python
from openai import OpenAI, Client
from vision_api_helpers import set_system_message, set_user_message
import os
from time import time, perf_counter
import pandas as pd
import json
from tqdm import tqdm


if __name__ == "__main__":
    out_file = 'openai_results_ship.csv'
    used_img_ids = []
    if os.path.exists(out_file): 
        print(f'WARNING outfile already exists {out_file}')
        df = pd.read_csv(out_file)
        used_img_ids = list(df['img_id'].unique())
        raise NameError
    # -- START -- set up run variables
    APIKEY='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'
    system_msg = """
    You are VisionPal, an AI assistant powered by GPT-4 with computer vision.
    AI knowledge cutoff: April 2023

    Built-in vision capabilities:
    - extract text from image
    - describe images
    - analyze image contents
    - logical problem-solving requiring machine vision
    """.strip()
    user_msgs = [
    """
    Describe every item in the image
    """.strip(),
    """
    Is there a ship in the image, if so how many?
    """.strip(),
    """
    Count the ships in the image
    """.strip()
    ]

    val_path = 'E:/datasets/DOTA_dataset_512/val_coco.json'
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    target_class = 3
    img_id_to_dict = { img_dict['id']:img_dict for img_dict in val_data['images']}
    cat_id_to_cat = { cat['id']:cat['name'] for cat in val_data['categories']}
    print(f'using {target_class}:{cat_id_to_cat[target_class]}')
    
    val_annots_df = pd.DataFrame(val_data['annotations'])

    plane_df = val_annots_df[val_annots_df['category_id']==target_class]
    print(f"found {len(plane_df['image_id'].unique())} images")
    #'''
    data_list = []
    if not os.path.exists(out_file): 
        df = pd.DataFrame(columns=['prompt', 'reply', 'time', 'img_id', 'img_path'])
    
    for img_id in tqdm(list(plane_df['image_id'].unique())):
        # skip loop if already used
        if img_id in used_img_ids: continue
        for user_msg in user_msgs:  
            img_path =img_id_to_dict[img_id]['path']
            image_paths = [img_path] 
            true_files = None  # you can give real names if using temp upload locations


            # Assemble the request parameters (all are dictionaries)
            system = set_system_message(system_msg)
            chat_hist = []  # list of more user/assistant items
            user = set_user_message(user_msg, image_paths, 1024)

            params = {  # dictionary format for ** unpacking
            "model": "gpt-4-turbo", "temperature": 0.5, "user": "my_customer",
            "max_tokens": 500, "top_p": 0.5, "stream": True,
            "messages": system + chat_hist + user,
            }
            
            start = perf_counter()
            try:
                client = Client(timeout=111,api_key=APIKEY)
                response = client.chat.completions.with_raw_response.create(**params)
                headers_dict = response.headers.items().mapping.copy()
                for key, value in headers_dict.items():  # set a variable for each header
                    locals()[f'headers_{key.replace("-", "_")}'] = value
            except Exception as e:
                print(f"Error during API call: {e}")
                response = None
                raise e

            if response is not None:
                try:
                    reply = ""
                    #print(f"---\nSENT:\n{user[0]['content'][0]}\n---")
                    for chunk_no, chunk in enumerate(response.parse()):
                        if chunk.choices[0].delta.content:
                            reply += chunk.choices[0].delta.content
                            #print(chunk.choices[0].delta.content, end="")
                    #data_list.append({'prompt':user_msg, 'reply':reply, 'time':perf_counter()-start, 'img_id':img_id,'img_path':img_path})
                    new_row_df = pd.DataFrame([{'prompt':user_msg, 'reply':reply, 'time':perf_counter()-start, 'img_id':img_id,'img_path':img_path}])
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    df.to_csv(out_file, index=False)
                except Exception as e:
                    print(f"Error during receive/parsing: {e}")
                    raise e

            #print(f"\n[elapsed: {perf_counter()-start:.2f} seconds]")
    #'''