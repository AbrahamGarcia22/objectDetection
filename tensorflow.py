import cv2
import time
import numpy as np

cap = cv2.VideoCapture('input/video_1.mp4')

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter('output/video_result_3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

with open('files/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
ssd_model = cv2.dnn.readNetFromTensorflow('files/frozen_inference_graph.pb', 'files/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt')

ln = ssd_model.getLayerNames()
ln = [ln[i-1] for i in ssd_model.getUnconnectedOutLayers()]
def preprocessing(image):
    R,G,B = cv2.split(image)
    eq_R = cv2.equalizeHist(R)
    eq_G = cv2.equalizeHist(G)
    eq_B = cv2.equalizeHist(B)
    image= cv2.merge([eq_R, eq_G, eq_B])
    image = cv2.GaussianBlur(image, (3,3), 0)
    return image
    
def detect(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = preprocessing(image)
    image_height, image_width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123))
    ssd_model.setInput(blob)

    output = ssd_model.forward()
    image_copy = np.copy(image)
    # print(output[0,0,:])
    # return
    # loop over each of the layer outputs
    # print(output)
    #print(layerInputs)
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .3:
            
            # get the class id
            #print(confidence)
            class_id = detection[1]
            final_prob = confidence * 100.
            # map the class id to the class
            class_name = class_names[int(class_id)-1]
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            text = f"{class_name}, {final_prob:.3f}"
            cv2.rectangle(image_copy, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
        # put the FPS text on top of the frame
            cv2.putText(image_copy, text, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
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
