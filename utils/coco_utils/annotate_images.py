from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import random
import os

BOX_COLOR = 255 #white

'''
    # draw text, half opacity
    d.text((10, 10), "Hello", font=fnt, fill=(255, 255, 255, 128))
    # draw text, full opacity
    d.text((10, 60), "World", font=fnt, fill=(255, 255, 255, 255))
'''
def generate_distinct_colors(num_colors,alpha=255):
    N = int(np.ceil(num_colors ** (1/3)))
    colors = []
    # Generate N evenly spaced points in RGB space
    for i in np.linspace(0, 1, num=N):
        for j in np.linspace(0, 1, num=N):
            for k in np.linspace(0, 1, num=N):
                if len(colors) < num_colors:
                    # Convert normalized (0.0-1.0) to 255 scale
                    color = (int(i * 255), int(j * 255), int(k * 255), alpha)
                    colors.append(color)
                else:
                    break
    
    return colors

def annotate_bbox(img, bbox, label, color=255, thickness=2, font_size=10, font_path=None):
    x_min, y_min, x_max, y_max = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    img_draw = ImageDraw.Draw(img)
    #print(color)
    #print(type(color))
    img_draw.rectangle(((x_min, y_min), (x_max, y_max)), outline=color, width=thickness)

    if font_path is None:
        font= ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)
    
    img_draw.text((x_min, y_min - int(1.3*font_size)), str(label), font=font, fill=color)

    return img

def convert_from_coco_to_xyxy(coco_bbox):
    return [coco_bbox[0], coco_bbox[1], coco_bbox[0]+coco_bbox[2], coco_bbox[1]+coco_bbox[3]]

def annotate_img(image, bboxes, category_ids, scores=None, category_id_to_name=None, 
                 colors_map=None, font_size=10, font_path=None, thickness=2,
                 bbox_format='coco', show_labels=True):
    img= image.copy()

    if bbox_format=='coco':
        bboxes = [convert_from_coco_to_xyxy(bbox) for bbox in bboxes]
    if scores is None:
        scores = ['']*len(bboxes)
    for bbox, category_id, score in zip(bboxes, category_ids,scores):
        if category_id_to_name is not None:
            class_name = category_id_to_name[category_id]
        else:
            class_name = category_id
        if show_labels:
            label = f'{class_name}|{score}'
        else:
            label="" 
        if colors_map is not None:
            try:
                if type(colors_map[category_id]) == list:
                    color = tuple(colors_map[category_id])
                else:
                    color = colors_map[category_id]
            except KeyError as e:
                color = BOX_COLOR
        else:
            color = BOX_COLOR
        
        img = annotate_bbox(img, bbox, label, color=color, font_size=font_size, 
                            font_path=font_path,thickness=thickness)
    return img
        
def annotate_coco_annotations(coco_file, img_root=None, sample_size=None, threshold=None, colors_map=None,
                             font_path=None, font_size=20, category_map=None, seed=None, thickness=2):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    if category_map is None:
        class_dict = { int(category['id']):category['name'] for category in coco_data['categories']}
    else:
        class_dict = category_map
    
    print('using labels')
    for id, name in class_dict.items():
        print(f'{id}|{name}')

    imgs = []
    img_ids = []
    img_names = []
    number_of_images = len(coco_data['images'])

    if sample_size is not None:
        if seed is not None:
            np.random.seed(seed)
        sample_set = np.random.choice(number_of_images, min(number_of_images, sample_size), replace=False)
    else:
        sample_set = range(number_of_images)

    for img_index in sample_set:
        image_dict = coco_data['images'][img_index]
        img_name = image_dict['file_name']
        try:
            if img_root is None:
                img = Image.open(image_dict['path']).convert('RGB')
            else:
                img = Image.open(f'{img_root}/{img_name}').convert('RGB')
        except:
            print(f'WARNING FILE NOT FOUND {img_name} | img id {image_dict["id"]}')
            continue

        img_id = image_dict['id']

        bboxes = []
        category_ids = []
        scores = []
        for annot in coco_data['annotations']:
            if annot['image_id'] == img_id:
                if threshold is None:
                    assert annot['category_id'] != 0, f'found annot[category_id] = {annot["category_id"]}'
                    category_ids.append(annot["category_id"])
                    bboxes.append(annot['bbox'])
                    try:
                        scores.append(f'{annot["score"]:.2f}')
                    except KeyError as e:
                        scores.append(1.0)
                elif annot['score'] > threshold:
                    category_ids.append(annot['category_id'])
                    bboxes.append(annot['bbox'])
                    scores.append(f'{annot["score"]:.2f}')
        if len(scores) != len(bboxes): scores = None
        annot_img = annotate_img(img, bboxes, category_ids, category_id_to_name=class_dict, 
                 colors_map=colors_map, font_size=font_size, font_path=font_path, thickness=thickness,
                 bbox_format='coco', scores=scores)
        imgs.append(annot_img)
        img_ids.append(img_id)
        img_names.append(img_name)
    return imgs, img_ids, img_names

                    
def get_random_coco_images(data, num_images=10):
    return random.sample(data['images'], num_images)

def get_coco_annotations_for_image(data, image_id):
    return [ann for ann in data['annotations'] if ann['image_id'] == image_id]

def annotate_segmentation_annotations(image, annotations, category_map=None, color_map=None,
                                      font_path=None, font_size=10, color='red', thickness=2 ):
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        segmentation = ann.get('segmentation', [])
        category=ann['category_id']
        if category_map is not None:
            label = category_map[category]
        else:
            label = category
        
        if color_map is not None:
            color = color_map[category]

        for seg in segmentation:
            points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            draw.polygon(points, outline=color, width=thickness)
        
            if font_path is None:
                font= ImageFont.load_default()
            else:
                font = ImageFont.truetype(font_path, font_size)
            xmin = min(point[0] for point in points)
            ymin = min(point[1] for point in points)
            draw.text((xmin, ymin - int(1.3*font_size)), str(label), font=font, fill=color)
    return image

def annotate_bbox_annotations(image, annotations, category_map=None, color_map=None,
                                      font_path=None, font_size=10, color='red', thickness=2 ):
    for ann in annotations:
        coco_box=ann['bbox']
        category=ann['category_id']
        if category_map is not None:
            label = category_map[category]
        else:
            label = category
        bbox = [coco_box[0], coco_box[1], coco_box[0]+coco_box[2], coco_box[1]+coco_box[3]]
        #print(coco_box)
        #print(bbox)
        if color_map is not None:
            color = color_map[category]
        image = annotate_bbox(image, bbox, label, color=color, thickness=thickness, font_size=font_size, font_path=font_path)
    return image

def annotate_coco_annotations_2(coco_file, sample_size, mode='bbox', img_root=None, colors_map=None,
                             font_path=None, font_size=20, category_map=None, seed=None, thickness=2):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    random_images = get_random_coco_images(coco_data, sample_size)
    images = []
    img_ids = []
    img_names = []
    for img_info in random_images:
        img_ids.append(img_info['id'])
        img_names.append(img_info['file_name'])
        if img_root is None:
            image_path = img_info['path']
        else:
            image_path = os.path.join(img_root, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        annotations = get_coco_annotations_for_image(coco_data, img_info['id'])
        if mode=='bbox':
            annotated_image = annotate_bbox_annotations(image, annotations, category_map=category_map, color_map=colors_map,
                                      font_path=font_path, font_size=font_size, color='red', thickness=thickness )
        elif mode=='seg':
            annotated_image = annotate_segmentation_annotations(image, annotations, category_map=category_map, color_map=colors_map,
                                      font_path=font_path, font_size=font_size, color='red', thickness=thickness )
        else:
            print('only accept bbox or seg modes')
            raise Exception
        
        images.append(annotated_image)
    return images, img_ids, img_names
