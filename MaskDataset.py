import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


def get_image(image: int):
    """
    Get a specific image of the Kaggle Mask dataset
    :param image: the id of the image between 0 and 852 (inclusive)
    :return: path of the stored image
    """
    image = max(0, min(image, 852))
    os.makedirs('kaggle/images', exist_ok=True)
    if not os.path.exists(os.path.join("kaggle", "images", "maksssksksss{}.png".format(image))):
        api.dataset_download_file(dataset='andrewmvd/face-mask-detection',
                                  file_name='images/maksssksksss{}.png'.format(image),
                                  path='./kaggle/images/')
    return os.path.join("kaggle", "images", "maksssksksss{}.png".format(image))


def get_dataset():
    """
    Download all the Kaggle Mask dataset
    :return: List of all the images' path
    """
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