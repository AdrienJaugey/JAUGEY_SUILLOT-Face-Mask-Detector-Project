import os
import random
import cv2
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import time
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils
import numpy as np
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

api = KaggleApi()
api.authenticate()


def get_kaggle_image(image: int):
    # The dataset we are using has images with number between 0 and 852
    image = max(0, min(image, 852))
    os.makedirs('kaggle/images', exist_ok=True)
    if not os.path.exists(os.path.join("kaggle", "images", "maksssksksss{}.png".format(image))):
        api.dataset_download_file(dataset='andrewmvd/face-mask-detection',
                                  file_name='images/maksssksksss{}.png'.format(image),
                                  path='./kaggle/images/')
    return os.path.join("kaggle", "images", "maksssksksss{}.png".format(image))


def get_kaggle_dataset():
    os.makedirs('kaggle/images', exist_ok=True)
    if os.path.exists(os.path.join('kaggle', 'images')):
        if len(os.listdir('kaggle/images')) != 853:
            api.dataset_download_files(dataset='andrewmvd/face-mask-detection', path='./')

            # Unzipping only images
            with zipfile.ZipFile('./face-mask-detection.zip', 'r') as archive:
                for file in archive.namelist():
                    if file.startswith('images/'):
                        archive.extract(file, './kaggle/')

    # Returning list of images path
    return [os.path.join('kaggle', "images", x) for x in os.listdir("kaggle/images")]


IMAGE_PATHS = get_kaggle_dataset()


LABEL_PATH = 'jaugey_suillot/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH, use_display_name=True)

PATH_TO_SAVED_MODEL = "jaugey_suillot/saved_model/"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


def load_image_into_numpy_array(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


# for image_path in random.sample(IMAGE_PATHS, 5):
for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    imageName = os.path.basename(image_path)
    cv2.imwrite("results/[OUT] " + imageName, image_np_with_detections)
    # plt.figure()
    # plt.imshow(image_np_with_detections)
    print('Done')
# plt.show()

print("end")
