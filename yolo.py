

# import necessary packages
import numpy as np
import time
import cv2
import os


# directory name of current file 
currentPath = os.path.dirname(os.path.realpath(__file__))


def yolo3_object_detector(imageFile):
    # load coco class labels of YOLO model was trained on
    labelsPath = os.path.sep.join([currentPath, "models", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # path to YOLO weights & model configuration
    weightsPath = os.path.sep.join([currentPath, "models", "yolov3.weights"])
    configPath = os.path.sep.join([currentPath, "models", "yolov3.cfg"])

    # load YOLO object detector trained on COCO dataset (80 classes)
    print("\t loading YOLO from disk ...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load input image and get its spatial dimensions
    image = cv2.imread(imageFile)
    (H, W) = image.shape[:2]

    # only get output layer names from YOLO
    layerNames = net.getLayerNames()
    layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # perform blob on input images and do forward pass of YOLO object detector
    # output bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layerNames)
    end = time.time()

    # show consuming time on YOLO
    print("\t YOLO takes {:.6f} seconds".format(end - start))

    # initialize lists of bbox, confidences, classID
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                topleft_X = int(centerX - (width / 2))
                topleft_Y = int(centerY - (height / 2))

                boxes.append([topleft_X, topleft_Y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw bbox & label in image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    imageFile = "/Users/taylorguo/Desktop/pics/butt/train/butt_0100.jpeg"
    yolo3_object_detector(imageFile)