<centre><h1>Fruit Basket - Tensorflow 2 Object Detection API</h1></centre>
<centre>![alt text](https://github.com/utkarsh-prakash/Classic-Fruit-Object-Detection/blob/main/images/inference.jpg?raw=true)

<h3>Here is the structure of every custom project folder:</h3>

- ```annotations ```: This folder will be used to store all ```*.csv``` files and the respective TensorFlow ```*.record files```, which contain the list of annotations for our dataset images.
- ```exported-models ```: This folder will be used to store exported versions of our trained model(s).
- ```images ```: This folder contains a copy of all the images in our dataset, as well as the respective ```*.xml``` files produced for each one, once labelImg is used to annotate objects.
    - ```images/train ```: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.
    - ```images/test ```: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.
- ```models ```: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file ```*.config```, as well as all files generated during the training and evaluation of our model.
- ```pre-trained-models ```: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.

<h3>Installing labelImg:</h3>

```bash
pip install labelImg
```
It can be run as follows:
```bash 
labelImg
# or
labelImg <PATH_TO_TF>/TensorFlow/workspace/training_demo/images
```
<h3>Label Map</h3>

Create json for total classes.<br>
Label map files have the extention ```.pbtxt``` and should be placed inside the ```training_demo/annotations``` folder.

<h3>Create Tensorflow Records</h3>

we have images in test/train directory in xml, which we want to convert into tf records using ```scripts>preprocessing>generate_tfrecord.py```
- Create train record
```bash
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record
```

- Create test record
```bash
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record
```
This will create 2 new files under the ```training_demo/annotations``` folder, named ```test.record``` and ```train.record```, respectively.

<h3> Pre-Trained Models</h3>

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
- download any and store in pre-trained-models.

<h3> Configure training pipeline </h3>

- Create directory for training job - for ex. ```ssd-resnet50```
- copy ```pipeline.config``` from respective pre-trained model to this training job directory.
- Change the pipline config file for values of
    - num_classes (line 3)
    - batch_size (line 131)
    - fine_tune_checkpoint to checkpoint file of model (line 161)
    - fine_tune_checkpoint_type to "detection" (line 167)
    - use_bfloat16 to false, as we are not training on tpu (line 168)
    - label_map path (line 172)
    - train record path (line 174)
    - label_map path (line 182)
    - test record path (line 186)
Note : line number are with respect to pipeline config file of ssd-resnet50-640x640 model

<h3> Training the model </h3>

- Get ```TensorFlow/models/research/object_detection/model_main_tf2.py``` to root directory.
- run the train command from root
- To run this training on google colab we will have to upload the whole project directory to google drive and mount the drive with colab.
- Note- For training on google colab, we will also have to upload the ```object_detection``` folder from ```Tensorflow > models > research``` to the root.
```bash
python model_main_tf2.py --model_dir=models/ssd-resnet50 --pipeline_config_path=models/ssd-resnet50/pipeline.config
```
- After the training is completed download the model from drive.

<h3> Evaluating a model with coco evaluation metrics </h3>

```bash
python model_main_tf2.py --model_dir=models/ssd-resnet50 --pipeline_config_path=models/ssd-resnet50/pipeline.config --checkpoint_dir=models/ssd-resnet50
```
This will give the coco evaluation on the test data. <br>
https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
```bash
Accumulating evaluation results...
DONE (t=0.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.793
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.926
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.865
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.838
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.832
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.879
```

<h3> TensorBoard monitoring </h3>

Training loss curve and evaluation metrics can be monitored with tensorBoard logs.
```bash
tensorboard --logdir=models/ssd-resnet50
```
<h3> Exporting the model </h3>

```bash
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/ssd-resnet50/pipeline.config --trained_checkpoint_dir models/ssd-resnet50/ --output_directory exported-models/model1
```
- We can do this on colab and download the exported model from drive ```exported-models > model1```
- Only the last checkpoint on model directory will be exported as model, rest all can be discarded.

<h3> Important Notes </h3>

- All the code files which are run for training/evaluation/exporting was modified to not use GPU on local machine. These code files were edited before being uploaded to drive.
- If training on local, pycoco and tensorflow will have numpy version conflict, due to which we will get two different errors if we try to make numpy compatible with either one of them. This version issue doesn't come up while training on colab, however Solution for local training is -
```bash
pip uninstall pycocotools 
pip install pycocotools-windows
```
- Error while exporting. Somehow the exporting code is not able to locate the checkpoint on the latest training. This error is originating from ```Tensorflow > models > research > object_detection > exporter_lib_v2.py```. We have to comment out ```status.assert_existing_objects_matched()``` on line 272. This change has to be done before running the setup file for installing research package on local. We can update the file before uploading object_detection folder to drive in case of colab training.