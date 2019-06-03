#!/usr/bin/env python

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import argparse
from PIL import Image
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
MODEL_IN_DIR = os.path.join("../weights")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco"))  # To find local version
#sys.path.insert(0,'samples/coco/')
import coco
import datetime
#%matplotlib inline 

parser = argparse.ArgumentParser()
parser.add_argument('--model',dest='model',default='mask_rcnn_coco.h5')
parser.add_argument('--input',dest='input',default='cam')
parser.add_argument('--media',dest='media',default='../lights_cluttered.webm')
parser.add_argument('--debug',dest='debug',action='store_true', default = False)
args = parser.parse_args()

VID_PATH = os.path.join(ROOT_DIR,"media",args.media)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "media",args.media)
print("MODEL USED :",args.model)
print("input: ",args.input)
print("media: ",args.media)
# Directory to save logs and trained model

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_IN_DIR, args.model)
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85

    if args.model == "jason_0040.h5":
        NUM_CLASSES = 1 + 5
    elif args.model == 'mask_rcnn_coco.h5':
        pass
    else:
        NUM_CLASSES = 1 + 6

config = InferenceConfig()
config.display()

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


###
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
if args.model == "mask_rcnn_coco.h5":
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
    #for idx, element in enumerate(class_names):
    #    print(idx, element)
elif args.model == "jason_0040.h5":
    class_names = ['BG', 'bypass-v', 'intake', 'ladderframe', 'pipe1', 'pipe2']
else:
    class_names = ['BG', 'bypass-r', 'bypass-v', 'intake', 'ladderframe', 'pipe1', 'pipe2']

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# for video and cap
loop = True

if args.input=='cam':
    capture = cv2.VideoCapture(0)
    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("Capture :",width,height,fps)
    # Define codec and create video writer
    file_name = "detection_capture{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
elif args.input=='video':
    capture = cv2.VideoCapture(VID_PATH)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15
    print("Video: ",width,height,fps)
    # Define codec and create video writer
    file_name = "detection_video{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
    
elif args.input == 'image':
    frame = skimage.io.imread(IMAGE_DIR)
    #print("image ;)",frame.shape)
    results = model.detect([frame], verbose=1) #1/0
    r = results[0]
    frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    file_name = "detection_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, frame)
    loop = False
else:
    print("what-----------------------------")
count = 0
if args.debug:
    dbug = 1
else:
    dbug = 0
print('Working frame by frame, saving to: ',file_name)
while loop:
    ret, frame = capture.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.detect([frame], verbose=dbug) #1/0
        r = results[0]
        frame = display_instances(
                frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if args.debug:
            cv2.imshow('frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        vwriter.write(frame)
        count += 1
    else:
        break

if not args.input == 'image':
    vwriter.release()
    capture.release()
    cv2.destroyAllWindows()

