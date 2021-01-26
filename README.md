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
 We use the [Efficientdet D1](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz) and [Efficientdet D0](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz) AIs from the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) that we have trained with our custom dataset.

 ## How to use
  1. Download or clone this repository
  2. Create a new Python environment (virtualenv, conda...)
  3. Using this environment, install pip packages using ```requirements.txt``` file
     * In your terminal, type ```python -m pip install -r requirements.txt```,
     * If you have a CUDA-Capable GPU, you can install CUDA and cuDNN for TensorFlow 2.3.1.
  4. ~~Build the application using the following command :~~
     * ~~```pyinstaller -F -c --add-data jaugey_suillot_d0_v1/;jaugey_suillot_d0_v1 --add-data jaugey_suillot_d1_v1/;jaugey_suillot_d1_v1 --add-data label_map.json;. -i ./FaceMaskDetector.ico FaceMaskDetector.py```~~
  5. Start the ~~```FaceMaskDetector.exe``` (located in the ```dist``` directory) using a command prompt~~ ```FaceMaskDetector.py``` script using same environment.

  ```
To start inference on the eval dataset :
  python FaceMaskDetector.py --eval [--version <version name>] [--minScore <score>] [--save] [--noviz]

To start inference on the camera stream :
  python FaceMaskDetector.py --inference [--version <version name>] [--minScore <score>] [--camera <camera name>]

To start inference on a single image :
  python FaceMaskDetector.py --inference [--version <version name>] [--minScore <score>] --file <file path> [--save] [--noviz]

Available arguments :
        --help, -h        Display this help.
        --eval, -e        Launch in Evaluation mode (run inferences on "images" folder).
        --inference, -i   Launch in Inference mode (Use camera flow as input).
        --version, -v     Select the model to use. Available models are : d0_v1, d1_v1
        --minScore, -m    Set the minimum score to display a detection (between 0 and 1 inclusive).
        --file, -f        Specify the input image for Inference mode instead of using camera.
        --save, -s        Images with detection results will be saved (not when using camera).
        --noviz           Image(s) will not be displayed.
        --camera, -c      Choosing camera/video stream to use. Default is 0.
  ```

 ## Project Members
|Name | Email adress |
|----------------|-----------------------------------|
|Adrien JAUGEY | adrien_jaugey@etu.u-bourgogne.fr |
|Bastien SUILLOT | bastien_suillot@etu.u-bourgogne.fr|
