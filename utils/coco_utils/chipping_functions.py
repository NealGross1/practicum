import json
import os
import numpy as np
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import shutil
import pandas as pd
import yaml
from coco_utils import create_df_from_coco
from shapely.geometry import Polygon
from shapely.wkt import loads

#def create_chipped_dataset_from_dataframes(train_df, val_df, destination, annotation_key, name_key, 
#                                           image_path_key, class_key, class_names, chip_size, chip_overlap, padding_threshold=0.9,
#                                           create_images=True, min_annoation_overlap=0.2, conversion='RGB', augmentations=None):
def chip_with_minimum_padding(img, chip_size, overlap=0.0):
    width, height = img.size
    pad_width = 0
    pad_height = 0
    if width < chip_size:
        print(f'WARNING ADDING MINIMUM PADDING FOR CHIPPING | image width = {width} chip_size = {chip_size}')
        pad_width = chip_size - width
        width = chip_size
    
    if height < chip_size:
        print(f'WARNING ADDING MINIMUM PADDING FOR CHIPPING | image height = {height} chip_size = {chip_size}')
        pad_height = chip_size - height
        height = chip_size
    
    if pad_height + pad_width > 0:
        padding = (0,0, pad_width, pad_height)
        img = ImageOps.expand(img, padding)
    
    #calculate the number of chips in x,y
    num_chips_x = int(np.ceil(width/(chip_size*(1-overlap))))
    num_chips_y = int(np.ceil(height/(chip_size*(1-overlap))))

    chips = []
    #generate chips through coordinates
    for j in range(num_chips_y):
        for i in range(num_chips_x):
            #calculate the coordinates
            x1 = int(i * chip_size *(1 - overlap))
            y1 = int(j * chip_size *(1 - overlap))
            x2 = x1 + chip_size
            y2 = y1 + chip_size
            # if chip extends a little bit over the edge shift it
            if x2 >= width:
                x1 = width - chip_size
                x2 = width
            
            if y2 >= height:
                y1 = height - chip_size
                y2 = height
            
            assert x1>=0, f'x1 < 0 | {x1}'
            assert y1>=0, f'y1 < 0 | {y1}'
            chip = img.crop((x1,y1,x2,y2))
            chip_x, chip_y = chip.size
            assert chip_x == chip_size and chip_y == chip_size, f'chip not created correctly | {x1},{y1},{x2},{y2}'
            chips.append((chip, (x1,y1)))

    return chips

def coco_to_shapely(segmentation):
    pixel_coords = segmentation[0]

    xy_coords = [(float(pixel_coords[i]), float(pixel_coords[i+1])) for i in range(0, len(pixel_coords), 2)]

    return Polygon(xy_coords)



def get_annotations_within_chip(img_df, chip_box, min_intersection=0.2, use_segmentation=True, target_classes=None ):
    chip_polygon = Polygon([(chip_box[0], chip_box[1]), (chip_box[2], chip_box[1]),
                            (chip_box[2], chip_box[3]), (chip_box[0], chip_box[3]),
                            (chip_box[0], chip_box[1])])
    annots_in_chip = []
    categories = []

    for index, row in img_df.iterrows():

        #skip data not in selected classes
        if target_classes is not None:
            if row['category_id'] not in target_classes: continue 

        # convert to chapely polygon to do intersection calc
        if use_segmentation:
            polygon = coco_to_shapely(row.segmentation)
        else:
            xy_bbox = row.xy_bbox
            polygon = Polygon([(xy_bbox[0], xy_bbox[1]), (xy_bbox[2], xy_bbox[1]),
                            (xy_bbox[2], xy_bbox[3]), (xy_bbox[0], xy_bbox[3]),
                            (xy_bbox[0], xy_bbox[1])])
        
        if chip_polygon.intersection(polygon):
            #calculate the intersection of the two polygons
            intersection_polygon = chip_polygon.intersection(polygon)
            intersection_area = intersection_polygon.area

            #calculate the area of the box polygon
            box_area = polygon.area

            #calculate the fract of intersection
            intersection_fraction = (intersection_area/box_area) 

            #add data if minimum intersection is met
            if intersection_fraction >= min_intersection:
                annots_in_chip.append(intersection_polygon)
                categories.append(row['category_id'])

    return categories, annots_in_chip

def polygon_to_bounding_coco_box(poly):
    min_x, min_y, max_x, max_y = poly.bounds

    width = max_x - min_x
    height = max_y - min_y

    return min_x, min_y, width, height

def chip_coco_dataset(src_coco_json, chip_size, out_dir, json_path, overlap=0.2, min_intersection=0.2, use_segmentation=True, 
                                use_empty=False, target_classes=None):
    with open(src_coco_json, 'r') as f:
        coco_json = json.load(f)

    coco_categories = coco_json['categories']
    #for category in coco_categories: print(category)
    coco_df = create_df_from_coco(src_coco_json)

    if target_classes is not None:
        coco_categories = [category for category in coco_categories if category['id'] in target_classes]
    
    os.makedirs(f'{out_dir}/images', exist_ok=True)

    coco_images = []
    coco_annots = []
    img_ind = 1
    annot_ind = 1
    for img_path in tqdm(coco_df['image_file_path'].unique(), desc='Processing Images', leave=False):
        img_df = coco_df[coco_df['image_file_path']==img_path]
        img = Image.open(img_path)
        chips = chip_with_minimum_padding(img, chip_size=chip_size, overlap=overlap)
        for chip_idx, (chip, corner) in tqdm(enumerate(chips), desc='Creating Chips'):
            chip_box = [corner[0], corner[1],corner[0] + chip_size - 1, corner[1] + chip_size - 1 ]
            categories, poly_annotations = get_annotations_within_chip(img_df, chip_box, 
                                                                       min_intersection=min_intersection,
                                                                       use_segmentation=True, target_classes=target_classes)
            for category_id, poly in zip(categories, poly_annotations):
                bbox = polygon_to_bounding_coco_box(poly)
                xy_poly = np.array(poly.exterior.coords).ravel().tolist()

                #shift bbox to fit new chip
                bbox = [bbox[0] - corner[0], bbox[1] - corner[1], bbox[2], bbox[3]]
                segmentation = []
                for i in range(0, len(xy_poly), 2):
                    segmentation.append(xy_poly[i] - corner[0])
                    segmentation.append(xy_poly[i+1] - corner[1])
                
                coco_annots.append({
                    'id': annot_ind, 
                    'image_id': img_ind,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': poly.area,
                    'iscrowd': 0,
                    'segmentation': [segmentation],
                    'polygon': poly.wkt
                })
                annot_ind+=1
            
            # save chips out
            if len(poly_annotations) > 0 or use_empty:
                src_name = img_path.split('/')[-1]
                base_name = src_name.split('.')[0]
                chip_name = f'{base_name}_{chip_idx}.png'
                chip_path = f'{out_dir}/images/{chip_name}'
                chip.save(chip_path)
                coco_images.append({
                    'id': img_ind,
                    'width': chip_size,
                    'height': chip_size,
                    'file_name': chip_name,
                    'licence': None,
                    'flicker_url': None,
                    'coco_url': None,
                    'date_captured': None,
                    'path': chip_path
                })
                img_ind +=1
    
    coco_json = {'categories': coco_categories,'images': coco_images, 'annotations':coco_annots}
    with open(json_path, 'w') as f:
        json.dump(coco_json, f, indent = 4)


def create_chipped_coco_dataset(coco_dir, chip_size, out_dir, train_json=None, val_json=None, overlap=0.2, min_intersection=0.2,
                               use_segmentation=True, use_empty=False, target_classes=None):
    if train_json is None:
        train_json=f'{coco_dir}/train_coco.json'
    if val_json is None:
        val_json=f'{coco_dir}/val_coco.json'
    
    os.makedirs(out_dir, exist_ok=True)

    for set_name, coco in [('train', train_json), ('val', val_json)]:
        print(f'chipping {set_name}')
        chip_coco_dataset(coco, chip_size, f'{out_dir}/{set_name}', f'{out_dir}/{set_name}_coco.json', 
                          overlap=overlap, min_intersection=min_intersection, use_segmentation=use_segmentation,
                          use_empty=use_empty, target_classes=target_classes)
         

if __name__=="__main__":
    dataset_dir = 'E:/datasets'
    dataset_name = 'DOTA_dataset'
    coco_dir = f'{dataset_dir}/{dataset_name}'
    chip_size = 512
    overlap = 0.2
    min_intersection = 0.2
    outdir = f'{dataset_dir}/{dataset_name}_{chip_size}'
    print('starting chipping process')
    create_chipped_coco_dataset(coco_dir, chip_size, outdir, overlap=overlap, min_intersection=min_intersection, use_segmentation=True, 
                                use_empty=False, target_classes=None)


