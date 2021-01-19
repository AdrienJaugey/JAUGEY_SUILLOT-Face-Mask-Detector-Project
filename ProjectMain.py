import getopt
import os
import sys
from time import time

import cv2
import MaskDataset
from TensorflowDetector import TensorflowDetector
from VideoCapture import VideoCapture


def inferenceOnFiles(detector:TensorflowDetector, saveImages=False, filePath=None, noviz=False, minScoreThreshold=None):
    """
    Run inference on the image given or the Kaggle dataset
    :param detector: the TensorflowDetector instance
    :param saveImages: if True, detected images will be saved
    :param filePath: if not None, will run inference on the image at this path instead of the Kaggle Dataset
    :param noviz: if True, images will not be shown
    :param minScoreThreshold: minimum score of detection to be shown
    :return: None
    """
    if minScoreThreshold is None:
        minScoreThreshold = 0.3
    if filePath is None:
        print("Retrieving Kaggle dataset... ", end="")
        IMAGE_PATHS = MaskDataset.get_dataset()
        print("Done")
    else:
        IMAGE_PATHS = [filePath]
    print("Press q to quit after first image was displayed")
    for image_path in IMAGE_PATHS:
        print("Running inference on {}... ".format(image_path), end="")
        image, result = detector.process(image_path)
        if filePath is None:  # Eval mode
            with open(os.path.join("results", "map", os.path.basename(image_path).split('.')[0] + '.txt'),
                      'w') as mapFile:
                mapFile.write(detector.getMapString(image, result))
        imageWithResult = detector.getResultImage(image, result, maxBoxesToDraw=100, minScoreThreshold=minScoreThreshold)
        print("Done")
        if saveImages:
            cv2.imwrite(os.path.join("results", os.path.basename(image_path)), imageWithResult)
        if not noviz:
            cv2.imshow('Detection (Press q to quit)', imageWithResult)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break


def inferenceOnCamera(detector:TensorflowDetector, minScoreThreshold=None):
    """
    Run inference on camera stream
    :param detector: the TensorflowDetector instance
    :param minScoreThreshold: minimum score of detection to be shown
    :return: None
    """
    if minScoreThreshold is None:
        minScoreThreshold = 0.5
    cap = VideoCapture(0)
    # cap = VideoCapture("http://192.168.1.19:8080/video")
    nb_avg = 0
    avg_fps = 0
    print("Press q to quit...")
    while True:
        frame = cap.read()
        if frame is None:
            continue
        start_time = time()
        image, result = detector.process(frame)
        imageWithResult = detector.getResultImage(image, result, maxBoxesToDraw=30, minScoreThreshold=minScoreThreshold)
        elapsed_time = time() - start_time
        fps = 1 / elapsed_time
        nb_avg += 1
        avg_fps = fps if nb_avg == 1 else ((avg_fps * (nb_avg - 1)) / nb_avg + (fps / nb_avg))

        imageWithResult = cv2.putText(imageWithResult, '{:.2f} FPS (avg : {:.2f})'.format(fps, avg_fps),
                                      (0, 15),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      (0, 0, 255),
                                      2)

        cv2.imshow('Camera (Press q to quit)', imageWithResult)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def displayHelp():
    """
    Display the help "screen"
    :return: None
    """
    print('ProjectMain.py -h')
    print('ProjectMain.py --eval (--minScore <score> ) (--save) (--noviz)')
    print('ProjectMain.py --inference (--minScore <score> ) (--file <file path> (--save) (--noviz))')
    print("\t--help, -h        Display this help.")
    print("\t--eval, -e        Launch in Evaluation mode (run inferences on kaggle dataset and save map files).")
    print("\t--inference, -i   Launch in Inference mode (Use camera flow as input).")
    print("\t--minScore, -m    Set the minimum score to display a detection (between 0 and 1 inclusive).")
    print("\t--file, -f        Specify the input image for Inference mode instead of using camera.")
    print("\t--save, -s        Images with detection results will be saved (not when using camera).")
    print("\t--noviz           Image(s) will not be displayed.")


def processArguments():
    """
    Process arguments of the command line to determine the execution mode
    :return: evalMode, inferenceMode, filePath, saveImages
    """
    evalMode = False
    inferenceMode = False
    filePath = None
    saveImages = False
    noviz = False
    minScore = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "heif:sm:", ["help", "eval", "inference", "file=", "save", "minScore=", "noviz"])
    except getopt.GetoptError:
        displayHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            displayHelp()
            sys.exit()
        elif opt in ("-e", "--eval"):
            evalMode = True
            if evalMode == inferenceMode:
                print("Cannot use evaluation mode and inference mode at the same time.\n")
                displayHelp()
                sys.exit(2)
        elif opt in ("-i", "--inference"):
            inferenceMode = True
            if evalMode == inferenceMode:
                print("Cannot use evaluation mode and inference mode at the same time.\n")
                displayHelp()
                sys.exit(2)
        elif opt in ("-f", "--file"):
            if not inferenceMode:
                print("Cannot use --file argument if not in inference mode.\n")
                displayHelp()
                sys.exit(2)
            filePath = arg
        elif opt in ("-s", "--save"):
            if inferenceMode:
                if filePath is None:
                    print("Cannot use --save argument.\n")
                    displayHelp()
                    sys.exit(2)
            saveImages = True
        elif opt in ("-m", "--minScore"):
            minScore = float(arg)
        elif opt == "--noviz":
            noviz = True
    return evalMode, inferenceMode, filePath, saveImages, noviz, minScore


if __name__ == "__main__":
    evalMode, inferenceMode, filePath, saveImages, noviz, minScore = processArguments()
    print("#### START ####")
    if not (evalMode or inferenceMode):
        inferenceMode = True
    print("Eval mode" if evalMode else "Inference mode")
    if minScore is not None:
        print("Detection minimum threshold set to {}".format(minScore))
    if filePath is not None:
        print("File Path = {}".format(filePath))
    if saveImages:
        print("Saving files")
    if noviz:
        print("No display of image(s)")
    print("Loading detector... ", end="")
    start_time = time()
    detector = TensorflowDetector(savedModelPath="jaugey_suillot_v1/saved_model/",
                                  labelMapPath="jaugey_suillot_v1/label_map.pbtxt")
    elapsed_time = time() - start_time
    print("Done ({:.2f} sec)\n".format(elapsed_time))
    os.makedirs(os.path.join("results", "map"), exist_ok=True)
    if (inferenceMode and filePath is not None) or evalMode:
        inferenceOnFiles(detector, saveImages=saveImages, filePath=filePath, noviz=noviz, minScoreThreshold=minScore)
    else:
        inferenceOnCamera(detector, minScoreThreshold=minScore)
    print("#### END ####")
