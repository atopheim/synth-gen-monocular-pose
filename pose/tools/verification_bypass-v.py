import sys
import pickle
sys.path.append('.')
sys.path.append('..')
from lib.networks.model_repository import *
from lib.utils.net_utils import smooth_l1_loss, load_model, compute_precision_recall
import torch
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.utils.evaluation_utils import pnp
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

with open('configs/homemade_train.json', 'r') as f:
    train_cfg = json.load(f)
train_cfg['model_name'] = '{}_{}'.format('intake', train_cfg['model_name'])

vote_num = 9

import cv2

class NetWrapper(nn.Module):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, image, mask, vertex, vertex_weights):
        seg_pred, vertex_pred = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)
        return seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall


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


def read_data(idx):
    import torchvision.transforms as transforms
    demo_dir = os.path.join(cfg.DATA_DIR,'demo','bypass-v')
    #source_dir = '/home/volvomlp2/python-envs/pvnet/data/HOMEMADE/renders/intake/validation'
    source_dir = os.path.join(demo_dir,'source')
    rgb = Image.open(os.path.join(source_dir, str(idx)+'.jpg'))
    mask = np.array(cv2.imread(os.path.join(source_dir, str(idx)+'_depth.png'))).astype(np.int32)[..., 0]
    #mask[mask != 0] = 1
    points_3d = np.loadtxt(os.path.join(demo_dir, 'bypass-v_points_3d.txt'))
    bb8_3d = np.loadtxt(os.path.join(cfg.HOMEMADE,'bypass-v','test_new_corners.txt'))#farthest
    pose = pickle.load(open(os.path.join(source_dir,str(idx)+'_RT.pkl'),'rb'))['RT']
    print("RT",pose)

    projector = Projector()
    points_2d = projector.project(points_3d, pose, 'blender')
    print("pts-2d",points_2d)
    vertex = compute_vertex(mask, points_2d)

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    rgb = transformer(rgb)
    #vertex = torch.tensor(vertex, dtype=torch.float32).permute(2, 0, 1)
    mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.int64)
    #vertex_weight = mask.unsqueeze(0).float()
    pose = torch.tensor(pose.astype(np.float32))
    #points_2d = torch.tensor(points_2d.astype(np.float32))
    data = (rgb, mask, pose)

    return data, bb8_3d


def visualize_mask(mask):
    plt.imshow(mask[0].cpu())
    plt.show()


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


def demo(idx):
    data, bb8_3d = read_data(idx)
    print("BB8_3D: ",bb8_3d)
    image, mask, pose = [d.unsqueeze(0).cuda() for d in data]
    projector = Projector()

    pose = pose[0].detach().cpu().numpy()
    bb8_2d_gt = projector.project(bb8_3d, pose, 'blender')
    print(bb8_2d_gt)
    
    image = imagenet_to_uint8(image.detach().cpu().numpy())[0]
    visualize_bounding_box(image[None, ...], bb8_2d_gt[None, None, ...])

if __name__ == "__main__":
    for idx in range(1,21):#(3482,3500):
        demo(idx)