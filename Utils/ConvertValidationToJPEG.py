import os
import cv2


def convert2JPEG(path, imageFile, outputPath):
    image = cv2.imread(os.path.join(path, imageFile))
    name = imageFile.split('.')[0]
    cv2.imwrite(os.path.join(outputPath, name + '.jpg'), image)


def updateXMLImageName(annotationFilePath):
    lines = []
    with open(annotationFilePath, 'r') as xml:
        lines = xml.readlines()
    lines[3] = lines[3].replace('png', 'jpg')
    with open(annotationFilePath, 'w') as xml:
        xml.writelines(lines)


def testXMLImageName(annotationFilePath):
    lines = []
    with open(annotationFilePath, 'r') as xml:
        lines = xml.readlines()
    assert '.jpg' in lines[3], "FileName not updated : " + annotationFilePath


DATASET_PATH = '..\\DATASETS\\VALIDATION'
IMAGES_PATH = os.path.join(DATASET_PATH, 'IMAGES')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'ANNOTATIONS')
OUTPUT_IMAGES_PATH = os.path.join(DATASET_PATH, 'JPEGImages')
os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)
imageList = os.listdir(IMAGES_PATH)
for idx, imageFile in enumerate(imageList):
    if idx % 25 == 0:
        print(str(idx), "/", str(len(imageList)), " ", imageFile)
    name = imageFile.split('.')[0]
    convert2JPEG(IMAGES_PATH, imageFile, OUTPUT_IMAGES_PATH)
    ANNOTATION_FILE_PATH = os.path.join(ANNOTATIONS_PATH, name + '.xml')
    updateXMLImageName(ANNOTATION_FILE_PATH)
    testXMLImageName(ANNOTATION_FILE_PATH)
