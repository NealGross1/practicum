# Project Overview

This document outlines the steps to recreate the results using the provided data files. Note that while the data CSVs are included, the image chipping process might yield different outputs from those originally used to generate them. For verification, refer to `explore_open_ai_results.ipynb` to check if the image IDs and descriptions in your dataset align with those used in this analysis.

## Environments Required

Three primary environments were used for various components of this project:

1. **ChatGPT4:** Requirements in `pip_freeze_openai_env.txt`
2. **BLIP2 VQA and Transformers:** Requirements in `pip_freeze_practicum_env.txt`
3. **YOLOv8:** Requirements in `pip_freeze_ultra_env.txt`

## Steps to Recreate Results

### 1. Obtain the Dataset
- Download the dataset from [DOTA](https://captain-whu.github.io/DOTA/dataset.html).
- Ensure training and validation data match the dataset provided at this link after chipping:
  - Training set: `DOTA_512/train_coco.json`
  - Validation set: `DOTA_512/val_coco.json`

### 2. Generate Chipped Dataset
- Use the chipping utility to create chipped images and corresponding COCO JSON files.
- Script: `utils/coco_utils/chipping_functions.py`

### 3. Create Augmented Dataset
- Generate a set of images with noise, blur, and pixel dropout augmentations.
- Notebook: `create_augmented_val_dataset.ipynb`

### 4. Generate Classification Results with BLIP2 VQA
- Run different prompts on each image and export results to CSV.
  - Script: `create_BLIP2_classification_responses.py` → Output: `prompt_results_classification.csv`
  - Script: `create_prompt_results_for_images_adv_a.py` → Output: `prompt_results_for_chips_adv_a.csv`

### 5. Generate Classification Results with ChatGPT4
- Execute prompts across images and save outputs.
  - Script: `PLANE_create_open_ai_results.py` → Output: `openai_results_plane.csv`
  - Script: `SHIP_create_open_ai_results.py` → Output: `openai_results_ship.csv`
  - Script: `AUG_create_open_ai_results.py` → Output: `openai_results_aug.csv`

### 6. Train YOLOv8 Model
- Train a YOLOv8 model for classification comparison.
- Script: `utils/yolo_utils/train_yolo_aug.py`
- Example training output is located in `YOLOv8/`

### 7. Evaluate Classification Performance
- Evaluate Visual Language Model (VLM) classifications.
  - Notebooks: `evaluate_yes_no_prompts.ipynb`, `BLIP2_augmentation_effects.ipynb`, `evaluate_chatgpt_classification.ipynb`

### 8. Generate YOLOv8 Metrics and Results
- Assess YOLOv8 performance metrics.
  - Notebooks and scripts: `utils/coco_utils/coco_eval.ipynb`, `create_yolo_coco_results_w_aug.py`, `eval_yolo_results.ipynb`
  - Outputs in `DOTA_512/`:
    - `coco_eval_by_class.json`, `coco_eval_results.json` (evaluation results)
    - `pr_curve_data.json` (PR curve data)
    - PR curve images in `.png` format

### 9. Extract Scene Descriptions with BLIP2 VQA
- Use prompts to retrieve scene details for each image.
  - Script: `create_prompt_results_for_annots.py` → Output: `prompt_results_annots.csv`
  - Script: `create_prompt_results_for_images.py` → Output: `prompt_results_for_chips.csv`

### 10. Scene Descriptions with ChatGPT4
- Scene description results are contained within the PLANE/SHIP/AUG CSV files.

### 11. Extract Top Words by Class from BLIP2 Scene Descriptions
- Analyze scene descriptions and identify top words by frequency for each class.
  - Notebook: `scene_description_analysis.ipynb` → Output: `top_word_counts_for_prompts_on_chips.csv`

### 12. Qualitative Search Analysis
- Examine scene descriptions to explore VLM strengths and weaknesses.
  - Notebooks: `explore_blip_results_adv_a.ipynb`, `explore_blip_results_annots.ipynb`, `explore_blip_results_chips.ipynb`, `explore_open_ai_results.ipynb`

### 13. Query Search
- Construct a DataFrame for text-based querying to identify:
  - False negatives
  - Specific objects
  - Notebooks: `search_queries.ipynb`, `setup_context_search.ipynb`

### 14. Embedding Search
- Generate VLM embeddings for object search using k-nearest neighbors with cosine similarity.
  - Notebook: `VLM_Embedding_Search.ipynb`

---

## Additional Resources

Additional files for ongoing work and exploration are located in the `random/` directory.
