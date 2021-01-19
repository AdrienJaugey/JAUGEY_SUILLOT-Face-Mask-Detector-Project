import os

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_image_into_numpy_array(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


class TensorflowDetector:
    """
    Encapsulation of a Tensorflow saved model to facilitate inferences and visualisation
    Based on : https://bit.ly/3p2iPDc
    """

    def __init__(self, savedModelPath: str, labelMapPath: str):
        self.__CATEGORY_INDEX__ = label_map_util.create_category_index_from_labelmap(labelMapPath,
                                                                                     use_display_name=True)
        self.__MODEL__ = tf.saved_model.load(savedModelPath)

    def process(self, image):
        """
        Run inference on an image
        :param image: path to the image to use or the image itself
        :return: The image and refactored dict from the inference
        """
        # Load image into a tensor
        assert image is not None, "Please provide at least a path to an image or an image directly."
        if type(image) is str:
            image_np = load_image_into_numpy_array(image)
        elif type(image) is np.ndarray:
            image_np = image.copy()
        else:
            assert False, "The type of image parameter does not match str or ndarray."
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Running the inference
        detections = self.__MODEL__(input_tensor)

        # Getting the number of detection (remove batch size from the original result)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Cast detection classes to correct format
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return image_np, detections

    def getMapString(self, image, results, minScoreThreshold=0.30):
        """
        Format result to the https://github.com/Cartucho/mAP#create-the-detection-results-files format
        :param image: the image to get the size from
        :param results: the detections of the image
        :param minScoreThreshold: Threshold to filter detections based on the score
        :return: The formatted result as string
        """
        res = ""
        for i in range(results['num_detections']):
            score = results['detection_scores'][i]
            if score >= minScoreThreshold:
                height, width, _ = image.shape
                yMin, xMin, yMax, xMax = tuple(results['detection_boxes'][i, :])
                yMin = int(yMin * height)
                xMin = int(xMin * width)
                yMax = int(yMax * height)
                xMax = int(xMax * width)
                classId = results['detection_classes'][i]
                className = self.__CATEGORY_INDEX__[classId]["name"]
                # <class_name> <confidence> <left> <top> <right> <bottom>
                res += "{}{} {:.6f} {} {} {} {}".format("" if res == "" else "\n", className, score,
                                                        xMin, yMax, xMax, yMin)
        return res

    def getResultImage(self, image, results: dict, maxBoxesToDraw=200, minScoreThreshold=0.30):
        """
        Draw results on the image
        :param image: the image to draw the results on
        :param results: the detection results of the image
        :param maxBoxesToDraw: maximum number of boxes to draw on the image
        :param minScoreThreshold: minimum score of the boxes to draw
        :return: The image with results
        """
        image_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            results['detection_boxes'],
            results['detection_classes'],
            results['detection_scores'],
            self.__CATEGORY_INDEX__,
            use_normalized_coordinates=True,
            max_boxes_to_draw=maxBoxesToDraw,
            min_score_thresh=minScoreThreshold,
            agnostic_mode=False,
            line_thickness=1)

        return image_with_detections
