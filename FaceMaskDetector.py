import os
import sys
import argparse
from time import time
import cv2
# import MaskDataset
from VideoCapture import VideoCapture
from TensorflowDetector import TensorflowDetector


def correct_path(value):
    try:
        path = os.path.normpath(value)
        return path
    except TypeError:
        raise argparse.ArgumentTypeError(f"{value} is not a correct path")


def existing_path(value):
    path = correct_path(value)
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{value} path does not exists")


def correct_score(value):
    try:
        value = float(value)
        if 0. <= float(value) <= 1.:
            return value
        else:
            raise argparse.ArgumentTypeError(f"{value} is not a number in [0; 1]")
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a number")


def camera_name(value):
    try:
        value = int(value)
        return value
    except ValueError:
        return value


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
        minScoreThreshold = 0.4
    if filePath is None:
        # print("Retrieving Kaggle dataset... ", end="", flush=True)
        # IMAGE_PATHS = MaskDataset.get_dataset()
        # print("Done")
        imageDir = "images"
        assert os.path.exists(imageDir), "Please put the evaluation images in a folder named \"images\""
        IMAGE_PATHS = [os.path.join('images', x) for x in os.listdir('images')]
    else:
        IMAGE_PATHS = [filePath]
    intentionnalStop = False
    time_sum = 0
    for image_path in IMAGE_PATHS:
        print("Running inference on {}... ".format(image_path), end="", flush=True)
        img_start_time = time()
        image, result = detector.process(image_path)
        image, mapText = detector.applyResults(image, result, maxBoxesToDraw=30,
                                               minScoreThreshold=minScoreThreshold,
                                               drawImage=saveImages or not noviz,
                                               mapText=filePath is None)
        time_sum += time() - img_start_time
        if filePath is None:  # Eval mode
            with open(os.path.join("results", "map", os.path.basename(image_path).split('.')[0] + '.txt'),
                      'w') as mapFile:
                mapFile.write(mapText)
        print("Done")
        if saveImages:
            cv2.imwrite(os.path.join("results", os.path.basename(image_path)), image)
        if not noviz:
            cv2.imshow('Detection (Press q to quit)', image)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                intentionnalStop = True
                break
    avg_time = round(time_sum / len(IMAGE_PATHS), 4)
    print("Avg Time per frame : {:.4f} sec".format(avg_time))
    print("Avg FPS : {:.4f}".format(1 / avg_time))
    if not noviz and not intentionnalStop:
        print("Inference(s) done ! Press q to quit...")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def inferenceOnCamera(detector: TensorflowDetector, cap: VideoCapture, minScoreThreshold=None):
    """
    Run inference on camera stream
    :param detector: the TensorflowDetector instance
    :param cap: the video capture
    :param minScoreThreshold: minimum score of detection to be shown
    :return: None
    """
    if minScoreThreshold is None:
        minScoreThreshold = 0.6
    nb_avg = 0
    avg_fps = 0
    frame_missing_countdown = 600
    while frame_missing_countdown > 0:
        frame = cap.read()
        if frame is None:
            frame_missing_countdown -= 1
            continue
        frame_missing_countdown = 600
        frame_start_time = time()
        image, result = detector.process(frame)
        imageWithResult, _ = detector.applyResults(image, result, maxBoxesToDraw=30,
                                                   minScoreThreshold=minScoreThreshold)
        frame_total_time = time() - frame_start_time
        fps = 1 / frame_total_time
        nb_avg += 1
        avg_fps = fps if nb_avg == 1 else ((avg_fps * (nb_avg - 1)) / nb_avg + (fps / nb_avg))

        imageWithResult = cv2.putText(imageWithResult, '{:.2f} FPS (avg : {:.2f})'.format(fps, avg_fps),
                                      (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Camera (Press q to quit)', imageWithResult)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def processArguments():
    """
    Process arguments of the command line to determine the execution mode
    :return: evalMode, inferenceMode, filePath, saveImages
    """
    parser = argparse.ArgumentParser("Face Mask Detector using EfficientDet D0 & D1. For more details see "
                                     "https://github.com/AdrienJaugey/JAUGEY_SUILLOT-Face-Mask-Detector-Project")
    parser.add_argument("-i", "--inference", dest="inference_mode", action="store_true",
                        help="Start the Face Mask Detector in Inference mode on camera input or given image. "
                             "Not adding this flag will start the Face Mask Detector in Evaluation mode (run"
                             " inferences on \"images\" folder).")
    parser.add_argument('-v', '--version', dest="model", type=str, default=list(AVAILABLE_MODELS.keys())[0],
                        choices=list(AVAILABLE_MODELS.keys()), help="Select the model to use.")
    parser.add_argument('-f', '--file', dest="file_path", default=None, type=existing_path,
                        help="Specify the input image for Inference mode instead of using camera. "
                             "Ignored if evaluation mode is enabled.")
    parser.add_argument('-s', "--save", dest="save_images", action="store_true",
                        help="Images with detection results will be saved (not when using camera).")
    parser.add_argument("-m", "--minScore", dest="min_score", type=correct_score, default=None,
                        help="Set the minimum score to display a detection (between 0 and 1 inclusive).")
    parser.add_argument("-nv", "--noviz", dest="no_viz", help="Image(s) will not be displayed.",
                        action="store_true")
    parser.add_argument("-c", "--camera", dest="camera_name", help="Choosing camera/video stream to use. Default is 0.",
                        type=camera_name, default=0)
    args = parser.parse_args()

    return (args.inference_mode, args.model, args.file_path,
            args.save_images, args.no_viz, args.min_score, args.camera_name)


def my_path(path_name):
    """Return the appropriate path for data files based on execution context"""
    # https://stackoverflow.com/questions/62518233/why-cant-my-pyinstaller-executable-access-data-files
    if getattr(sys, 'frozen', False):
        # running in a bundle
        return os.path.join(sys._MEIPASS, path_name)
    else:
        # running live
        return path_name


AVAILABLE_MODELS = {
    "d0_v1": {'path': "jaugey_suillot_d0_v1",
              "description": "EfficientDet D0 (Batch of 8, 53100 steps, LR = .04, "
                             "Warmup LR = .0001, 5310 warmup steps)"},
    "d1_v1": {'path': "jaugey_suillot_d1_v1",
              "description": "EfficientDet D1 (Batch of 4, 53100 steps, LR = .002, "
                             "Warmup LR = .0001, 12000 warmup steps)"}
}

if __name__ == "__main__":
    inferenceMode, model, filePath, saveImages, noviz, minScore, cameraName = processArguments()
    print("#### START ####")
    print(f"Mode : {'Inference' if inferenceMode else 'Evaluation'}")
    print(f"Model used : {AVAILABLE_MODELS[model]['description']}")
    if minScore is not None:
        print(f"Detection minimum threshold : {minScore:.2%}")
    if filePath is not None:
        print(f"File Path : {filePath}")
    if saveImages:
        print("Image(s) will be saved")
    if noviz:
        print("Image(s) will not be displayed")
    if inferenceMode and filePath is None:
        camera = VideoCapture(cameraName)
        if cameraName != 0:
            print(f"Camera to use : {cameraName}")
    savedModelPath = my_path(os.path.join(AVAILABLE_MODELS[model]["path"], 'saved_model'))
    labelMapPath = my_path('label_map.json')
    print("\nLoading detector... ", end="", flush=True)
    start_time = time()
    detector = TensorflowDetector(savedModelPath=savedModelPath, labelMapPath=labelMapPath)
    elapsed_time = time() - start_time
    print(f"Done ({elapsed_time:.2f} sec)\n")
    if (inferenceMode and filePath is not None) or not inferenceMode:
        if not inferenceMode or saveImages:
            os.makedirs(os.path.join("results", "map"), exist_ok=True)
        inferenceOnFiles(detector, saveImages=saveImages, filePath=filePath, noviz=noviz, minScoreThreshold=minScore)
    else:
        inferenceOnCamera(detector, cap=camera, minScoreThreshold=minScore)
    print("#### END ####")
