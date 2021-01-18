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
 
 ## Project Members
 |Name | Email adress |
|----------------|-----------------------------------|
|Adrien JAUGEY | adrien_jaugey@etu.u-bourgogne.fr |
|Bastien SUILLOT | bastien_suillot@etu.u-bourgogne.fr|
