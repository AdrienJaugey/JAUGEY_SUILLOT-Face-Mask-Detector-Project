import os
import json
import warnings
import cv2
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)


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
        with open(labelMapPath, 'r') as file:
            self.__CATEGORY_INDEX__ = {int(key): value for key, value in json.load(file).items()}
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

    def applyResults(self, image, results: dict, maxBoxesToDraw=200, minScoreThreshold=0.3, drawImage=True, mapText=False):
        """
        Draw results on the image
        :param image: the image to draw the results on
        :param results: the detection results of the image
        :param maxBoxesToDraw: maximum number of boxes to draw on the image
        :param minScoreThreshold: minimum score of the boxes to draw
        :param drawImage: if True, will draw bboxes on the image
        :param mapText: if True, will return the content of the map txt file as a string
        :return: The image with results if enabled else None, the map file as string if enabled
        """
        image_with_detections = image.copy() if drawImage else None
        resMapText = "" if mapText else None
        boxesCount = 0
        for idx in range(results['num_detections']):
            score = results['detection_scores'][idx]
            if score >= minScoreThreshold and boxesCount < maxBoxesToDraw:
                height, width, _ = image.shape
                yMin, xMin, yMax, xMax = tuple(results['detection_boxes'][idx, :])
                yMin = int(yMin * height)
                xMin = int(xMin * width)
                yMax = int(yMax * height)
                xMax = int(xMax * width)
                classId = results['detection_classes'][idx]
                className = self.__CATEGORY_INDEX__[classId]["name"]
                if mapText:
                    # <class_name> <confidence> <left> <top> <right> <bottom>
                    resMapText += "{}{} {:.6f} {} {} {} {}".format("" if resMapText == "" else "\n", className, score,
                                                                   xMin, yMax, xMax, yMin)
                if drawImage:
                    color = tuple(self.__CATEGORY_INDEX__[classId]["color"])
                    image_with_detections = cv2.rectangle(image_with_detections, (xMin, yMin), (xMax, yMax), color, 3)
                    scoreText = '{}: {:.0%}'.format(className, score)
                    image_with_detections[yMin - 12:yMin + 2, xMin:xMin + 9 * len(scoreText), :] = color
                    image_with_detections = cv2.putText(image_with_detections, scoreText, (xMin, yMin),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                boxesCount += 1
        return image_with_detections, resMapText
