import os
import cv2
import MaskDataset
from TensorflowDetector import TensorflowDetector

print("#### START ####")
print("Retrieving Kaggle dataset... ", end="")
IMAGE_PATHS = MaskDataset.get_dataset()
print("Done")
print("Loading detector... ", end="")
detector = TensorflowDetector(savedModelPath="jaugey_suillot/saved_model/",
                              labelMapPath="jaugey_suillot/label_map.pbtxt")
print("Done\n")

print("Press q to quit after first image was displayed")
for image_path in IMAGE_PATHS:
    print("Running inference on {}... ".format(image_path), end="")
    image, result = detector.process(image_path)
    imageWithResult = detector.getResultImage(image, result, maxBoxesToDraw=200, minScoreThreshold=.3)
    print("Done")

    cv2.imshow('Detection', imageWithResult)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# cap = cv2.VideoCapture(0)
# while True:
#     # Find haar cascade to draw bounding box around face
#     ret, frame = cap.read()
#     if not ret:
#         break

print("#### END ####")
