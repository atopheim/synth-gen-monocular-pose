import sys
import time
sys.path.append('.')
sys.path.append('..')
from lib.networks.model_repository import *
#from lib.networks.vgg16_convs import VGG16Convs
from lib.utils.arg_utils import args
from lib.utils.net_utils import smooth_l1_loss, load_model, compute_precision_recall
import torch
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.utils.evaluation_utils_homemade import pnp
from lib.utils.draw_utils import imagenet_to_uint8, visualize_bounding_box
from lib.utils.base_utils import Projector
import json
from lib.utils.config import cfg
from torch.nn import DataParallel
from torch import nn, optim
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime

#with open(args.cfg_file, 'r') as f:
#    train_cfg = json.load(f)
#train_cfg['model_name'] = '{}_{}'.format(args.homemade_cls, train_cfg['model_name'])

with open(args.cfg_file, 'r') as f:
    train_cfg = json.load(f)
    print(args)
train_cfg['model_name'] = '{}_{}'.format(args.homemade_cls, train_cfg['model_name'])

vote_num = 9

print("Type of program :",args.type)
print("Media: ",args.media)

class NetWrapper(nn.Module):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduce=False)
    
    def forward(self, image):
        seg_pred, vertex_pred = self.net(image)
        return seg_pred, vertex_pred
"""
    def forward(self, image, mask, vertex, vertex_weights):
        seg_pred, vertex_pred = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
        loss_vertex = smooth_l1_mooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)
        return seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall
"""

class EvalWrapper(nn.Module):
    def forward(self, seg_pred, vertex_pred, use_argmax=True):
        vertex_pred = vertex_pred.permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex_pred.shape
        vertex_pred = vertex_pred.view(b, h, w, vn_2 // 2, 2)
        if use_argmax:
            mask = torch.argmax(seg_pred, 1)
        else:
            mask = seg_pred
        return ransac_voting_layer_v3(mask, vertex_pred, 512, inlier_thresh=0.99)

def compute_vertex(mask, points_2d):
    num_keypoints = points_2d.shape[0]
    h, w = mask.shape
    m = points_2d.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = xy[:, None, :] * np.ones(shape=[1, num_keypoints, 1])
    vertex = points_2d[None, :, :2] - vertex
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm < 1e-3] += 1e-3
    vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    return np.reshape(vertex_out, [h, w, m * 2])

def visualize_mask(mask,count):
    plt.imshow(mask[0].cpu())
    plt.show(block=False)
    plt.pause(0.1)
    plt.save('{}_mask.png'.format(count))
    plt.close()
    #plt.show()
    

def visualize_vertex(vertex, vertex_weights):
    vertex = vertex * vertex_weights
    for i in range(9):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        ax1.imshow(vertex[0, 2*i].cpu())
        ax2.imshow(vertex[0, 2*i+1].cpu())
        plt.show()


def visualize_hypothesis(image, seg_pred, vertex_pred, corner_target):
    from lib.ransac_voting_gpu_layer.ransac_voting_gpu import generate_hypothesis

    vertex_pred = vertex_pred.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex_pred.shape
    vertex_pred = vertex_pred.view(b, h, w, vn_2 // 2, 2)
    mask = torch.argmax(seg_pred, 1)
    hyp, hyp_counts = generate_hypothesis(mask, vertex_pred, 1024, inlier_thresh=0.99)

    image = imagenet_to_uint8(image.detach().cpu().numpy())
    hyp = hyp.detach().cpu().numpy()
    hyp_counts = hyp_counts.detach().cpu().numpy()

    from lib.utils.draw_utils import visualize_hypothesis
    visualize_hypothesis(image, hyp, hyp_counts, corner_target)


def visualize_voting_ellipse(image, seg_pred, vertex_pred, corner_target):
    from lib.ransac_voting_gpu_layer.ransac_voting_gpu import estimate_voting_distribution_with_mean

    vertex_pred = vertex_pred.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex_pred.shape
    vertex_pred = vertex_pred.view(b, h, w, vn_2//2, 2)
    mask = torch.argmax(seg_pred, 1)
    mean = ransac_voting_layer_v3(mask, vertex_pred, 512, inlier_thresh=0.99)
    mean, var = estimate_voting_distribution_with_mean(mask, vertex_pred, mean)

    image = imagenet_to_uint8(image.detach().cpu().numpy())
    mean = mean.detach().cpu().numpy()
    var = var.detach().cpu().numpy()
    corner_target = corner_target.detach().cpu().numpy()

    from lib.utils.draw_utils import visualize_voting_ellipse
    visualize_voting_ellipse(image, mean, var, corner_target)

def inference(input_image,count=0):
    c_timer = time.time()
    rgb = input_image
    if args.input != 'image':
        color = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
        rgb = color
    pre_start = time.time()
    print(pre_start- c_timer, "s BGR2RGB")
    #rgb = Image.open(input_image)
    #print(rgb.shape)
    start = time.time()
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    rgb = transformer(rgb)
    rgb = rgb.unsqueeze(0).cuda()
    seg_pred, vertex_pred = net(rgb)
    eval_net = DataParallel(EvalWrapper().cuda())
    corner_pred = eval_net(seg_pred, vertex_pred).cpu().detach().numpy()[0]
    end = time.time()
    print(end-start, "s - to go from image to corner prediction")
    image = imagenet_to_uint8(rgb.detach().cpu().numpy())[0]
    pose_pred = pnp(points_3d, corner_pred, camera_matrix)
    projector = Projector()
    bb8_2d_pred = projector.project(bb8_3d, pose_pred, 'logitech')
    end_= time.time()
    print(end_-end, "s - to project the corners and show the result")
    seg_mask = torch.argmax(seg_pred, 1)
    if args.debug:
        visualize_mask(seg_mask,count)
        pose_test = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0.3],
                    [0, 0, 1, 1.2]])
        print(pose_pred)
        #print(pose_test)
        bb8_2d_gt = projector.project(bb8_3d, pose_test, 'logitech')
    if pose_pred[2][3] < 0.4:
        if pose_pred[2][3] > -0.4:
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
            rgb = rgb.astype(np.uint8)
            _, ax = plt.subplots(1)
            ax.imshow(cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB))
            #plt.show()
            plt.savefig('temp{}.png'.format(count))
            plt.close()
            
            print("image was culled due to pose being unreasonable")
    else:
        visualize_bounding_box(image[None, ...], bb8_2d_pred[None, None, ...],save=True,count=count)#,bb8_2d_gt[None, None, ...])
    
    
if __name__ == "__main__":
    import torchvision.transforms as transforms
    
    net = Resnet18_8s(ver_dim=vote_num * 2, seg_dim=2)
    net = NetWrapper(net).cuda()
    net = DataParallel(net)
    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
    #load model
    model_dir = os.path.join(cfg.MODEL_DIR, "{}_demo".format(args.homemade_cls))
    load_model(net.module.net, optimizer, model_dir, args.load_epoch)
    demo_dir = os.path.join(cfg.DATA_DIR, 'demo','{}'.format(args.homemade_cls))
    #load bb8
    bb8_3d = np.loadtxt(os.path.join(cfg.HOMEMADE,'{}'.format(args.homemade_cls),'corners.txt'))#, delimiter=" ", usecols=range(3))
    points_3d = np.loadtxt(os.path.join(cfg.HOMEMADE, '{}'.format(args.homemade_cls),'{}_points_3d.txt'.format(args.homemade_cls)))
    
    camera_matrix = np.array([[542.9039, 0., 323.9369],
                              [0., 529.055241, 239.57233],
                              [0., 0., 1.]])                              

    """
    inference(os.path.join(demo_dir,'source_real','intake_logi_3.jpg'))
    """ 

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
        VID_PATH = os.path.join('/home/volvomlp2/Videos/Webcam',args.media)
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
        import skimage.io
        IMG_PATH = os.path.join('/home/volvomlp2/Pictures/Webcam',args.media)
        frame = skimage.io.imread(IMG_PATH)
        print("image ;)",frame.shape)
        inference(frame)
        file_name = "detection_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #skimage.io.imsave(file_name, frame)
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
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inference(frame,count)
            #img = cv2.imread('temp{}.png'.format(count))
            #cv2.imshow("image",img)
            #if cv2.waitKey(10) & 0xFF == ord('q'):
            #    break
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #vwriter.write(img)
            
            print(count)
            count += 1
        else:
            break
    if args.input != 'image':
        vwriter.release()
        capture.release()
        cv2.destroyAllWindows()