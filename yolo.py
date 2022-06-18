import cv2
import time
import numpy as np

cap = cv2.VideoCapture('input/video_1.mp4')

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter('output/video_result_3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

with open('files/classes_coco_cafe.txt', 'r') as f:
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
yolo_model = cv2.dnn.readNetFromDarknet('files/yolov3.cfg', 'files/yolov3.weights')

ln = yolo_model.getLayerNames()
ln = [ln[i-1] for i in yolo_model.getUnconnectedOutLayers()]
def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(image)
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    image = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2RGB)
    # R,G,B = cv2.split(image)
    # eq_R = cv2.equalizeHist(R)
    # eq_G = cv2.equalizeHist(G)
    # eq_B = cv2.equalizeHist(B)
    # image= cv2.merge([eq_R, eq_G, eq_B])
    # image = cv2.GaussianBlur(image, (3,3), 0)
    return image
    
def detect(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = preprocessing(image)
    image_height, image_width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), crop=False, interpolation=cv2.INTER_NEAREST)
    yolo_model.setInput(blob)

    layerOutputs = yolo_model.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    percentages = []

    image_copy = np.copy(image)
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
 
            # map the max confidence to the class label names

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.6:
                # probs = np.exp(classID) / np.sum(np.exp(classID))
                final_prob = confidence * 100.
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([image_width, image_height,image_width, image_height])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                percentages.append(final_prob)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)
            out_name = class_names[classIDs[i]]
            percent = percentages[i]
            text = f"{out_name}, {percent:.3f}"
            # text = f"{class_names[classIDs[i]]}"
            cv2.putText(image_copy, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image_copy
# def prepocess(img):
#     img = cv2.eq
    
# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image = frame
        p = detect(image)
        cv2.imshow('image', p)
        out.write(p)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
