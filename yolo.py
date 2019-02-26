

# import necessary packages
import os, time, cv2, csv
import numpy as np
from imutils import paths

# directory name of current file
current_path = os.path.dirname(os.path.realpath(__file__))

# path to YOLO weights & model configuration
weights_path = os.path.sep.join([current_path, "models", "yolov3-tiny.weights"])
config_path = os.path.sep.join([current_path, "models", "yolov3-tiny.cfg"])

# load coco class labels of YOLO model was trained on
labels_path = os.path.sep.join([current_path, "models", "coco.names"])

# weightsPath = os.path.sep.join([currentPath, "models", "yolov3.weights"])
# configPath = os.path.sep.join([currentPath, "models", "yolov3.cfg"])

class yolov3_detector:

    def __init__(self):
        self.config = config_path
        self.weights = weights_path
        self.labels = labels_path

    def load_net_weights(self):
        # load YOLO object detector trained on COCO dataset (80 classes)
        print(" - Loading YOLO from disk ...")
        net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        return net

    def object_detector(self, net, imageFile):

        LABELS = open(self.labels).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

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
        print(" - YOLO takes {:.3f} seconds".format(end - start))

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

        result_coordinate = {}
        result_coordinate["image"] = imageFile
        result_coordinate["num_objects"] = len(idxs)
        obj_dict = {}
        obj_name_dict = {}

        if len(idxs) > 0:
            for i in idxs.flatten():
                obj_item = {i: (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])}
                obj_dict.update(obj_item)
                obj_name_item = {i: LABELS[classIDs[i]]}
                obj_name_dict.update(obj_name_item)

                # (x, y) = (boxes[i][0], boxes[i][1])
                # (w, h) = (boxes[i][2], boxes[i][3])
                # # draw bbox & label in image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        result_coordinate["objects"] = obj_dict
        result_coordinate["names"] = obj_name_dict

        # print(result_coordinate)
        return result_coordinate

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)

if __name__ == "__main__":
    # imageFile = "/Users/taylorguo/Desktop/pics/butt/train/butt_0100.jpeg"
    # imageFile = "/Users/taylorguo/Documents/projects_2019/yolov3-darknet/darknet/data/dog.jpg"
    image_folder = "/Users/taylorguo/Documents/github/yolov3/data/samples/mt/products"

    detector = yolov3_detector()
    net = detector.load_net_weights()

    results = []
    for i in paths.list_images(image_folder):
        result = detector.object_detector(net, i)
        results.append(result)

    with open("test_result_1.csv", "a", newline="") as f:
        csv_writer = csv.writer(f)
        for i in results:
            csv_writer.writerow([i["image"], i["num_objects"], i["objects"], i["names"]])
