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


def show(pred_cur, img, trajs):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    def compute_color_for_labels(label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)
    

    det_bboxes = pred_cur[:, 2:6]
    # ious = box_iou(torch.from_numpy(det_bboxes[:, :4]), torch.from_numpy(gt_bboxes[:, :4]))
    # max_value, max_index = ious.max(-1)

    for bbox in pred_cur:
        frame_id, id, x_min, y_min, x_max, y_max, mode, *_ = bbox

        rgb = compute_color_for_labels(id)
        img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), rgb, 2)
        if y_min < 30:
            img = cv2.putText(img, '%d_%0.0f'%(id, mode), (int(x_min), int(y_max)), cv2.FONT_HERSHEY_PLAIN, 2, rgb, 2, cv2.LINE_AA)
        else:
            img = cv2.putText(img, '%d_%0.0f'%(id, mode), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_PLAIN, 2, rgb, 2, cv2.LINE_AA)
        # else:
        #     img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        #     if y_min < 30:
        #         img = cv2.putText(img, '%d'%(id), (int(x_min), int(y_max)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
        #     else:
        #         img = cv2.putText(img, '%d'%(id), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

        #     frame_id, id, x_min, y_min, x_max, y_max, *_ = gt_cur[index]
        #     img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255), 2)
        #     if y_min < 30:
        #         img = cv2.putText(img, '%d'%(id), (int(x_min), int(y_max)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        #     else:
        #         img = cv2.putText(img, '%d'%(id), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # for ith, bbox in enumerate(gt_cur):
    #     if ith not in max_index:
    #         frame_id, id, x_min, y_min, x_max, y_max, *_ = bbox
    #         img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 255), 2)
    #         if y_min < 30:
    #             img = cv2.putText(img, '%d'%(id), (int(x_min), int(y_max)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    #         else:
    #             img = cv2.putText(img, '%d'%(id), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)


    return img


if __name__ == "__main__":
    gt_src = "datasets/dancetrack/test"
    ours = "tmp" # preds
    
    seqs = os.listdir(gt_src)
    for seq in ['dancetrack0085']: # seqs:
        pred = np.loadtxt(os.path.join(ours, "{}.txt".format(seq)), delimiter=",")
        pred[:, [4,5]] += pred[:, [2,3]]

        os.makedirs('tmp/%s'%(seq), exist_ok=True)

        frame_id = 0
        trajs = {}
        while True:
            frame_id+=1
            print(frame_id)
            img_p = os.path.join(gt_src, seq, "img1/{:08d}.jpg".format(frame_id))
            if not os.path.exists(img_p):break
            pred_cur = pred[pred[:, 0] == frame_id]

            img = cv2.imread(img_p)

            img = show(pred_cur, img, trajs)

            img = cv2.putText(img, str(frame_id), (int(40), int(40)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow('result', img)
            # cv2.waitKey(0)
            cv2.imwrite('tmp/%s/%08d.jpg'%(seq, frame_id), img)

            if frame_id == 1:
                fps = 16 
                size = (img.shape[1], img.shape[0]) 
                videowriter = cv2.VideoWriter(os.path.join("tmp", "{}/tmp.avi".format(seq)), cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

            videowriter.write(img)
        
        videowriter.release()