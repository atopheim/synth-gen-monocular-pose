import os
import sys
import json
import datetime
import numpy as np
import cv2
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import random
import skimage.io

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#   export PYTHONWARNINGS="ignore"

# Root directory of the project
ROOT_DIR = os.getcwd()
MASKRCNN_DIR = '/media/volvomlp2/b70eabd5-2b0b-4fbe-8230-7e839a24249a/2jonas/repositories/Mask_RCNN'
# Import Mask RCNN
sys.path.append(MASKRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

from pycocotools import mask as maskUtils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class EagleEyeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "eagleeye-synthesis"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  #

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Learning rate
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.8

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    


############################################################
#  Dataset
############################################################

class EagleEyeDataset(utils.Dataset):

    def load_eagleeye(self, dataset_dir, subset, classIds=None):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # # Train or validation dataset?
        # assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        eagleeye = json.load(open(os.path.join(dataset_dir, "coco.json")))
        print('classIds', classIds)

        # add classes
        for category in eagleeye["categories"]:

            # if classIds is not None and category['id'] not in classIds:
            #     continue

            self.add_class("eagleeye", category["id"], category["name"])

        # Add images
        for imageInfo in eagleeye["images"]:

            annotations = []
            for anno in eagleeye["annotations"]:

                if classIds is not None and anno['category_id'] not in classIds:
                    continue

                if anno['image_id'] == imageInfo["id"]:
                    annotations.append(anno)

            if not annotations:
                continue

            self.add_image(
                "eagleeye", image_id=imageInfo["id"],
                path=os.path.join(dataset_dir, imageInfo["file_name"]),
                width=imageInfo["width"],
                height=imageInfo["height"],
                annotations=annotations)




    # def load_image(self, image_id):
    #     image_info = self.image_info[image_id]
    #     return cv2.imread(image_info['path'])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """


        # If not a eagleeye dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "eagleeye":
            return super(self.__class__, self).load_mask(image_id)


        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "eagleeye.{}".format(annotation['category_id']))
            segmentations = annotation['segmentation']
            pts = [np
                    .array(contour)
                    .reshape(-1, 2)
                    .round()
                    .astype(int)
                for contour in segmentations
            ]
            m = np.zeros((image_info["height"], image_info["width"]), np.uint8)
            m = cv2.fillPoly(m, pts, 1)

            instance_masks.append(m)
            class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__, self).load_mask(image_id)



    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["file_name"]

############################################################
#  Visualization
############################################################


def make_visuals(model, classNames = None,
                 imagePath = None, videoPath = None, outPath = None):
    assert imagePath or videoPath
    assert outPath

    classNames = ['BG', 'bypass-r', 'bypass-v', 'intake', 'ladderframe', 'pipe1', 'pipe2']

    # Image or video?
    if imagePath:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(imagePath))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # Create visual and save image
        visualize_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    classNames, r['scores'], making_image=True, file_name = outPath)

    elif videoPath:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(videoPath)
        # width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = 1600
        height = 1600
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        vwriter = cv2.VideoWriter(outPath,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        # For video, we wish classes keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(classNames))

        while success:
            print("frame: ", count)
            # Read next image
            plt.clf()
            plt.close()
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                # frame = color_splash(image, r['masks'])

                frame = visualize_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                     classNames, r['scores'], colors=colors, making_video=True)
                frame = cv2.resize(frame, (width, height))
                # Add image to video writer
                vwriter.write(frame)
                count += 1
        vwriter.release()
    print("Saved to ", outPath)



# visualize information to file
def visualize_instances(image, boxes, masks, class_ids, class_names, scores = None, ax= None,
                   file_name = None, colors = None,
                   making_video=False, making_image = False, open_cv = False
                   ):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]


    if not making_video:
        # Generate random colors
        colors = visualize.random_colors(N)

    fig, ax = get_ax()

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if not making_video:
            color = colors[i]
        else:
            # all objects of a class get same color
            class_id = class_ids[i]
            color = colors[class_id-1]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]

        # show bbox
        p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        try:
            label = class_names[class_id]
        except IndexError:
            label = 'label_error'

        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label

        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))

    # To transform the drawn figure into ndarray X
    fig.canvas.draw()
    string = fig.canvas.tostring_rgb()
    masked = masked_image.astype(np.uint8)
    X = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    X = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))


    # open cv's RGB style: BGR
    # if open_cv:
    X = X[..., ::-1]


    if making_video:
        plt.close()
        # return masked_image.astype(np.uint8)
        # cv2.imshow('frame', X)
        # cv2.waitKey(1)
        return X

    elif making_image:
        # plt.savefig(file_name)
        cv2.imwrite(file_name, X)


############################################################
#  Evaluation
############################################################
#


def evaluate(model, dataset, render=False, output='', imgIds=None, classIds=None):

    t_prediction = 0
    t_start = time.time()
    accu = {
        "rois": [],
        "class_ids": [],
        "scores": [],
        "masks": [],
    }

    accu_gt = {
        "ids": [],
        "boxes": [],
        "masks": []
    }

    results = []
    evals = []
    if imgIds is None:
        imgIds = dataset.image_ids

    # imgIds = [imgIds[0]]
    for i, image_id in enumerate(imgIds):

        print("Evaluating image", str(i), 'of', str(len(imgIds)))

        # Load image
        # sometimes skimage reads jpg as PMOImage file, rewrite image using OpenCV
        image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # filter classes
        if classIds is not None:
            idx = []
            for i, classId in enumerate(r['class_ids']):
                if classId not in classIds:
                    idx.append(i)
            r['rois'] = np.delete(r['rois'], idx, 0)
            r['masks'] = np.delete(r['masks'], idx, -1)
            r['class_ids'] = np.delete(r['class_ids'], idx, 0)
            r['scores'] = np.delete(r['scores'], idx, 0)

        accu['rois'].append(r['rois'])
        accu['masks'].append(r['masks'])
        accu['class_ids'].append(r['class_ids'])
        accu['scores'].append((r['scores']))

        accu_gt['ids'].append(gt_class_ids)
        accu_gt['boxes'].append(gt_bboxes)
        accu_gt['masks'].append(gt_masks)

        render = True
        # save rendering
        if render:
            filepath = os.path.join(os.path.dirname(output), os.path.splitext(os.path.basename(output))[0],
                                    str(image_id) + '.jpg')

            if not os.path.isdir(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))

            print("exporting to ", filepath)
            # visualize ground truth
            # visualize_instances(image, gt_bbox, gt_mask, gt_class_id,
            #                     dataset.class_names, file_name = filepath, making_image=True)
            visualize_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], file_name=filepath, making_image=True)

    if len(accu['scores']) > 0:

        AP, precisions, recalls, overlaps = utils.compute_ap(np.concatenate(accu_gt['boxes'], 0), np.concatenate(accu_gt['ids'], 0), np.concatenate(accu_gt['masks'], -1),
                                            np.concatenate(accu['rois'], 0), np.concatenate(accu['class_ids'], 0), np.concatenate(accu['scores'],0), np.concatenate(accu['masks'], -1))


        evals = {
            'mAP': AP,
            'precisions': np.round(precisions, 3).tolist(),
            'recalls': np.round(recalls,3).tolist(),
            # 'overlaps': np.round(overlaps, 3).tolist()
            'prediction_time': t_prediction,
            'prediction_time_average:': t_prediction / len(dataset.image_ids)
        }
    else:
        evals = {
            'mAP': 0,
            'precisions': [],
            'recalls': [],
            # 'overlaps': np.round(overlaps, 3).tolist()
            'prediction_time': 0,
            'prediction_time_average:': 0
        }

    with open(output, 'w') as outfile:
        json.dump(evals, outfile)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(dataset.image_ids)))
    print("Total time: ", time.time() - t_start)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return fig, ax


############################################################
#  Training
############################################################
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = EagleEyeDataset()
    dataset_train.load_eagleeye(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = EagleEyeDataset()
    dataset_val.load_eagleeye(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=EagleEyeConfig.LEARNING_RATE,
                epochs=80,
                layers='heads')

    # # Training - Stage 2
    # # Finetune layers from ResNet stage 4 and up
    # print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                 learning_rate=EagleEyeConfig.LEARNING_RATE,
                 epochs=120,
                 layers='4+')
    #
    # # Training - Stage 3
    # # Fine tune all layers
    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=160,
    #             layers='all')





if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='EagleEye Synthesis Training and Evaluation System')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--output', required=False, metavar="Path of output file")

    parser.add_argument('--classID', required=False, metavar="ClassID to be isolated")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
        assert args.weights, "Argument --weights is required for training (pretrained weights)"
    if args.command == "eval":
        assert args.dataset, "Argument --dataset is required for evaluation"
        assert args.weights, "Argument --weights is required for evaluation (trained weights)"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = EagleEyeConfig()
    else:
        class InferenceConfig(EagleEyeConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.command == "train":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "eval":
        if args.classID:
            classIDs = [int(args.classID)]
        else:
            classIDs = None

        # Validation dataset
        dataset = EagleEyeDataset()
        dataset.load_eagleeye(args.dataset, 'eval', classIds=classIDs)
        dataset.prepare()

        evaluate(model, dataset, output=args.output, classIds=classIDs)
    elif args.command == "splash":
        make_visuals(model, imagePath=args.image,
                                videoPath=args.video, outPath=args.output)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'eval'".format(args.command))
