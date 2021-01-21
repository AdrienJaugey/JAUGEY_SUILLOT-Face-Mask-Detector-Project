import os
import sys
import getopt
from time import time
import cv2
import MaskDataset
from VideoCapture import VideoCapture
from TensorflowDetector import TensorflowDetector


def inferenceOnFiles(detector: TensorflowDetector, saveImages=False, filePath=None, noviz=False,
                     minScoreThreshold=None):
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
        print("Retrieving Kaggle dataset... ", end="", flush=True)
        IMAGE_PATHS = MaskDataset.get_dataset()
        print("Done")
    else:
        IMAGE_PATHS = [filePath]
    print("Press q to quit after first image was displayed")
    for image_path in IMAGE_PATHS:
        print("Running inference on {}... ".format(image_path), end="", flush=True)
        image, result = detector.process(image_path)
        image, mapText = detector.applyResults(image, result, maxBoxesToDraw=30,
                                               minScoreThreshold=minScoreThreshold,
                                               mapText=filePath is None)
        if filePath is None:  # Eval mode
            with open(os.path.join("results", "map", os.path.basename(image_path).split('.')[0] + '.txt'),
                      'w') as mapFile:
                mapFile.write(mapText)
        print("Done")
        if saveImages:
            cv2.imwrite(os.path.join("results", os.path.basename(image_path)), image)
        if not noviz:
            cv2.imshow('Detection (Press q to quit)', image)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break


def inferenceOnCamera(detector: TensorflowDetector, cap: VideoCapture, minScoreThreshold=None):
    """
    Run inference on camera stream
    :param detector: the TensorflowDetector instance
    :param minScoreThreshold: minimum score of detection to be shown
    :return: None
    """
    if minScoreThreshold is None:
        minScoreThreshold = 0.5
    nb_avg = 0
    avg_fps = 0
    print("Press q to quit...")
    while True:
        frame = cap.read()
        if frame is None:
            continue
        start_time = time()
        image, result = detector.process(frame)
        imageWithResult, _ = detector.applyResults(image, result, maxBoxesToDraw=30,
                                                   minScoreThreshold=minScoreThreshold)
        elapsed_time = time() - start_time
        fps = 1 / elapsed_time
        nb_avg += 1
        avg_fps = fps if nb_avg == 1 else ((avg_fps * (nb_avg - 1)) / nb_avg + (fps / nb_avg))

        imageWithResult = cv2.putText(imageWithResult, '{:.2f} FPS (avg : {:.2f})'.format(fps, avg_fps),
                                      (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Camera (Press q to quit)', imageWithResult)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def displayHelp():
    """
    Display the help "screen"
    :return: None
    """
    print('To start inference on the eval dataset :')
    print('FaceMaskDetector.exe --eval (--minScore <score> ) (--save) (--noviz)\n')
    print('To start inference on the camera stream :')
    print('FaceMaskDetector.exe --inference (--minScore <score> ) (--camera <camera name>)\n')
    print('To start inference on a single image :')
    print('FaceMaskDetector.exe --inference (--minScore <score> ) --file <file path> (--save) (--noviz)\n')
    print("Available arguments :")
    print("\t--help, -h        Display this help.")
    print("\t--eval, -e        Launch in Evaluation mode (run inferences on kaggle dataset and save map files).")
    print("\t--inference, -i   Launch in Inference mode (Use camera flow as input).")
    print("\t--minScore, -m    Set the minimum score to display a detection (between 0 and 1 inclusive).")
    print("\t--file, -f        Specify the input image for Inference mode instead of using camera.")
    print("\t--save, -s        Images with detection results will be saved (not when using camera).")
    print("\t--noviz           Image(s) will not be displayed.")
    print("\t--camera, -c      Choosing camera/video stream to use. Default is 0.")


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
    cameraName = 0
    try:
        opts, args = getopt.getopt(sys.argv[1:], "heif:sm:c:",
                                   ["help", "eval", "inference", "file=", "save", "minScore=", "noviz", "camera="])
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
        elif opt in ("-c", "--camera"):
            cameraName = arg
    return evalMode, inferenceMode, filePath, saveImages, noviz, minScore, cameraName


def my_path(path_name):
    """Return the appropriate path for data files based on execution context"""
    # https://stackoverflow.com/questions/62518233/why-cant-my-pyinstaller-executable-access-data-files
    if getattr(sys, 'frozen', False):
        # running in a bundle
        return os.path.join(sys._MEIPASS, path_name)
    else:
        # running live
        return path_name


if __name__ == "__main__":
    evalMode, inferenceMode, filePath, saveImages, noviz, minScore, cameraName = processArguments()
    print("#### START ####")
    if not (evalMode or inferenceMode):
        inferenceMode = True
    print("Mode : ", "Evaluation" if evalMode else "Inference")
    if minScore is not None:
        print("Detection minimum threshold : {:.2%}".format(minScore))
    if filePath is not None:
        print("File Path : {}".format(filePath))
    if saveImages:
        print("Image(s) will be saved")
    if noviz:
        print("Image(s) will be displayed")
    if inferenceMode and filePath is None:
        camera = VideoCapture(cameraName)
        if cameraName != 0:
            print("Camera to use : {}".format(cameraName))

    savedModelPath = my_path(os.path.join('jaugey_suillot_v1', 'saved_model'))
    labelMapPath = my_path('label_map.json')
    print("\nLoading detector... ", end="", flush=True)
    start_time = time()
    detector = TensorflowDetector(savedModelPath=savedModelPath, labelMapPath=labelMapPath)
    elapsed_time = time() - start_time
    print("Done ({:.2f} sec)\n".format(elapsed_time))
    if (inferenceMode and filePath is not None) or evalMode:
        if evalMode or saveImages:
            os.makedirs(os.path.join("results", "map"), exist_ok=True)
        inferenceOnFiles(detector, saveImages=saveImages, filePath=filePath, noviz=noviz, minScoreThreshold=minScore)
    else:
        inferenceOnCamera(detector, cap=camera, minScoreThreshold=minScore)
    print("#### END ####")
    input("Pres ENTER to exit...")
