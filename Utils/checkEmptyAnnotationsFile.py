import os
import lxml

ANNOTATIONS_DIR_PATH = "..\\MaskTrainDataset\\Annotations"
LOG_FILE_PATH = "..\\MaskTrainDataset\\xml_status.log"
ONLY_BAD = True
xmlList = os.listdir(ANNOTATIONS_DIR_PATH)

badXML = []
with open(LOG_FILE_PATH, 'w') as log:
    for file in xmlList:
        filePath = os.path.join(ANNOTATIONS_DIR_PATH, file)
        with open(filePath, 'r') as xmlFile:
            lines = xmlFile.readlines()
        objectFound = False
        for line in lines:
            objectFound = objectFound or (line.find("object") != -1)
            if objectFound:
                break
        status = "OK" if objectFound else "FAIL"
        if not objectFound:
            badXML.append(file.replace(".xml", ""))
        if not ONLY_BAD:
            log.write("{} - {}\n".format(file, status))
    log.write("\nBad XMLs :")
    for bad in badXML:
        log.write("\n{}".format(bad))
