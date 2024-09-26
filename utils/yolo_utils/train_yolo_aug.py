from ultralytics import YOLO
import albumentations

if __name__ == '__main__':
  ### IMPORTANT ###
  ### hyper paramteres for aug and model ###
  model_version = 8
  workers = 11
  image_size = 512
  batch_size = 16
  epochs = 100
  device = 0
  use_amp = True
  hyp=dict()
  hyp['hsv_h'] = 0.015 # image HSV-hue aug
  hyp['hsv_s'] = 0.1 # image HSV-sat aug
  hyp['hsv_v'] = 0.1 # iamge HSV-val aug
  hyp['degrees'] = 15 # image rotation
  hyp['translate'] = 0.1 # +/- fraction translation
  hyp['scale'] = 0.1 # default = 0.5 image scale +/- gain
  hyp['shear'] = 0.0 # image shear +/- degree
  hyp['perspective'] = 0.1 # +/- fraction range 0 - 0.001
  hyp['flipud'] = 0.2 # image flip up down prob
  hyp['fliplr'] = 0.2 # image flip left right prob
  hyp['mosaic'] = 1.0 # mosaic prob
  hyp['mixup'] = 0.0 # image mixe up prob
  hyp['copy_paste'] = 0.0 # segment copy-paste prob


  ### IMPORTANT ###
  ### Model Save ###
  root = 'E:/datasets'
  data_name = 'DOTA_dataset_512'
  data_yml = f'{root}/{data_name}/yolo_config.yaml'

  for model_size in ['m']:
      model = YOLO(f'yolov{model_version}{model_size}.pt')

      experiment_name = f'E:/models/{data_name}_yolov{model_version}{model_size}_bb_aug'
      results = model.train(data=data_yml,
                            epochs=epochs,
                            batch=batch_size,
                            workers=workers,
                            patience=50,
                            imgsz=image_size,
                            device=device,
                            save_period=50,
                            single_cls=False,
                            name=experiment_name,
                            augment=True,
                            amp=use_amp,
                            hsv_h= hyp['hsv_h'],
                            hsv_s=hyp['hsv_s'], 
                            hsv_v=hyp['hsv_v'],
                            degrees=hyp['degrees'], 
                            translate=hyp['translate'],
                            scale=hyp['scale'],
                            shear=hyp['shear'], 
                            #perspective=hyp['perspective'], broken
                            flipud=hyp['flipud'], 
                            fliplr=hyp['fliplr'], 
                            mosaic=hyp['mosaic'], 
                            mixup=hyp['mixup'], 
                            copy_paste=hyp['copy_paste'],
                          ) 
      

    
    