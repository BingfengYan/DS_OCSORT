import os
import cv2
import numpy as np

def box_iou(box1, box2):
   
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = np.clip((np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])), 0, 1920).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


root = '/home/hadoop-vacv/yanfeng/data/dancetrack/train'

videos_name = os.listdir(root)

for v_name in videos_name:

    if 'dancetrack' not in v_name: continue

    imgs_path = os.listdir(os.path.join(root, v_name, 'img1'))
    labels = np.loadtxt(os.path.join(root, v_name, 'gt/gt.txt'), delimiter=",").reshape(-1, 9)
    labels[:, [4,5]] += labels[:, [2,3]]
    for ith, img_name in enumerate(imgs_path):

        print(ith, img_name)

        img = cv2.imread(os.path.join(root, v_name, 'img1', img_name))
        img_show = img.copy()

        frame_id = int(os.path.splitext(img_name)[0])
        gts = labels[labels[:, 0]==frame_id]
        if len(gts) > 1:
            bboxes = gts[:, 2:6]
            ious = box_iou(bboxes, bboxes)

            for jth, (gt, iou) in enumerate(zip(gts, ious)):
                img_crop = img[int(gt[3]): int(gt[5]), int(gt[2]): int(gt[4])]
                if min(img_crop.shape) == 0 : continue
                max_idx = iou.argsort()[-2]
                if iou[max_idx] > 0:
                    if gt[5] >  gts[max_idx, 5]:
                        label_crop = 1
                    else:
                        label_crop = 2
                else:
                    label_crop = 0

                save_p = os.path.join(root, 'forw_back', str(label_crop), '%08d_%02d.jpg'%(frame_id, jth))
                os.makedirs(os.path.dirname(save_p), exist_ok=True)
                cv2.imwrite(save_p, img_crop)

                img_show = cv2.rectangle(img_show, (int(gt[2]), int(gt[3])), (int(gt[4]), int(gt[5])), (255, 255, 255), 2)
                img_show = cv2.putText(img_show, '%d'%(label_crop), (int(gt[2]), int(gt[3])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # cv2.imshow('result', img_show)
        # cv2.waitKey(0)
        # cv2.imwrite('tmp.png', img_show)