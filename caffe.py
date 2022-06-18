import cv2
import time
import numpy as np
import sys


if len(sys.argv) != 3:
    print("Error!\nUso: pyhton3 caffe.py [input_video_path] [output_video_path]")
    exit(1)
video_in = sys.argv[1]
video_out = sys.argv[2]

cap = cv2.VideoCapture(video_in)

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

with open('files/classes_coco_cafe.txt', 'r') as f:
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
caffe_model = cv2.dnn.readNetFromCaffe('files/deploy.prototxt', 'files/VGG_coco_SSD_300x300_iter_400000.caffemodel')

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
    # image = cv2.pyrDown(image)
    return image
    
def detect(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocessing(image)
    image_height, image_width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123))
    caffe_model.setInput(blob)

    output = caffe_model.forward()
    image_copy = np.copy(image)
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .5:
            
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
    
#detect objects in each frame of the video
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
