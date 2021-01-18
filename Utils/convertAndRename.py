import os
import cv2

INPUT_IMAGES_PATH = 'AvecMask'
OUTPUT_DIR = 'out'
RENAME_TO = 'JAUGEY_SUILLOT_'

if not os.path.exists(INPUT_IMAGES_PATH):
    print(INPUT_IMAGES_PATH, " not found")
    exit(1)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for i, imagePath in enumerate(os.listdir(INPUT_IMAGES_PATH)):
    name = imagePath.split('.')[0]
    outputName = RENAME_TO + '{:0>3d}'.format(i + 1) + '.jpg'
    outputPath = os.path.join(OUTPUT_DIR, outputName)
    print(i, ' : ', imagePath, ' --> ', outputPath)
    image = cv2.imread(os.path.join(INPUT_IMAGES_PATH, imagePath))
    cv2.imwrite(outputPath, image)
