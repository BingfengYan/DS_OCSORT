"""
    This script is to draw trajectory prediction as in Fig.6 of the paper
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np 
import os
import torch


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)



if __name__ == "__main__":

    result_iou = []
    hists = None
    gt_src = "datasets/dancetrack/val"

    seqs = os.listdir(gt_src)
    for seq in seqs:

        gt = np.loadtxt(os.path.join(gt_src, seq, "gt/gt.txt"), delimiter=",")

        gt[:, [4,5]] += gt[:, [2,3]]

        ious = box_iou(torch.from_numpy(gt[:, 2:6]), torch.from_numpy(gt[:, 2:6]))

        # result_iou.append(ious.cpu().numpy().reshape(-1))
        hist, bins = np.histogram(ious.cpu().numpy().reshape(-1), bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999, 1.0]) 
        if hists is None:
            hists=hist
        else:
            hists+=hist

    # result_iou = np.hstack(result_iou)

    # hist,bins = np.histogram(result_iou, bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999, 1.0]) 

    # plt.hist(result_iou, bins =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999, 1.0]) 
    # plt.title("histogram") 
    # plt.savefig('tmp.png')
    # # plt.show()

    print(hists)
