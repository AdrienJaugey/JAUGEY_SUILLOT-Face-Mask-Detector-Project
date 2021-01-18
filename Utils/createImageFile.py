import os
DATASET_PATH = '..\\MaskTrainDataset\\'
dirs = ['TRAIN', 'VALIDATION']
dirs = {'TRAIN': 'train', 'VALIDATION': 'val'}
TXT = "images.txt"

images = os.listdir(os.path.join(DATASET_PATH, 'JPEGImages'))
with open(os.path.join(DATASET_PATH, TXT), 'w') as txtFile:
    for image in images:
        txtFile.write(image.split('.')[0] + "\n")
