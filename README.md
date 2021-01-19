# Facemask Detector Project
This project is the final class project of an AI classes during last year of IT Engineering School.

## Goal
The main goal of this project is to develop and train an AI (completely or by using existing ones) that will detect faces with or without masks, and also improperly worn masks.

## Datasets
 * The evaluation dataset is the [Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection) dataset by andrewmvd on Kaggle.
 * The dataset used to train this detector is a custom dataset made by all the groups working on this project.

We will use the same classes as the Kaggle dataset :
 * ```with_mask``` : the person is wearing a face mask properly,
 * ```mask_weared_incorrect``` : the person is wearing a face mask improperly,
 * ```without_mask``` : the person is not wearing a face mask.
 
 ## AI
 We use the [Efficientdet D1](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz) AI from the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) that we have trained with our custom dataset.
 
 ## How to use
  1. Download or clone this repository
  2. Download or clone the [tensorflow/models repository](https://github.com/tensorflow/models)
  3. Move "models" directory inside this repo directory
  4. Install the OD API
     * [Tutorial](https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85), you can skip the docker part,
     * If you have a CUDA-Capable GPU, you can install CUDA and cuDNN for your TensorFlow version.
  5. Using the same environment/python as the OD API, start the ```ProjectMain.py```
  
  ```
  python ProjectMain.py [-h | --eval (--save) (--noviz)| --inference (--file <file path> (--save) (--noviz))]
      --help, -h        Display this help.
      --eval, -e        Launch in Evaluation mode (run inferences on kaggle dataset and save map files).
      --inference, -i   Launch in Inference mode (Use camera flow as input).
      --file, -f        Specify the input image for Inference mode instead of using camera.
      --save, -s        Images with detection results will be saved (not when using camera).
      --noviz           Image(s) will not be displayed.
  ```
 
 ## Project Members
 |Name | Email adress |
|----------------|-----------------------------------|
|Adrien JAUGEY | adrien_jaugey@etu.u-bourgogne.fr |
|Bastien SUILLOT | bastien_suillot@etu.u-bourgogne.fr|
