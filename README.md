TensorFlow-based object detection codes.

## Implemented models and data I/Os

### Model
- [YOLOV2](https://arxiv.org/abs/1612.08242) model.

### Data I/O
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset.

## Prepare .tfrecord data
This repo only assumes .tfrecord format data.  
The .tfrecord data should be created by `scripts/write_tfrecord_${data_name}.py`.  

## Training
You can train model by `scripts/trainer_${model_name}_${data_name}.py`.  
Example parameters are defined in `scripts/params/train_params_${model_name}_${data_name}.yaml`.  
