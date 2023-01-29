"""
https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun
"""
import os
import numpy as np
import json
import cv2


DATA_PATH = 'datasets/MMPTracking'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
# SPLITS = ['train', 'val', 'test']
SPLITS = ["validation"]

cam_parm = {'cafe_shop': 4,
            'industry_safety': 4,
            'lobby': 4,
            'office': 5,
            'retail': 6
            }

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:

        data_path = os.path.join(DATA_PATH, split)
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'dancer'}]}
        seqs_main = os.listdir(os.path.join(data_path, 'images'))
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq_main in sorted(seqs_main):
            seqs = os.listdir(os.path.join(data_path, 'images', seq_main))
            for seq in sorted(seqs):
                if '.DS_Store' in seq or '.ipy' in seq:
                    continue

                img_path = os.path.join(data_path, 'images', seq_main, seq)
                images = os.listdir(img_path)
                num_images = len([image for image in images if 'jpg' in image])  # half and half
                i=0
                for cam_id in range(1, cam_parm[seq[:-2]]+1):
                    video_cnt += 1  # video sequence number.
                    out['videos'].append({'id': video_cnt, 'file_name': '%s_%s_%d'%(seq_main, seq, cam_id)})
                    frame_id = 0
                    while True:
                        img_p = os.path.join(data_path, 'images', seq_main, '{}/rgb_{:05d}_{:d}.jpg'.format(seq, frame_id, cam_id))
                        if not os.path.exists(img_p): break

                        img = cv2.imread(img_p)
                        height, width = img.shape[:2]
                        image_info = {'file_name': '{}/{}/{}/rgb_{:05d}_{:d}.jpg'.format('images', seq_main, seq, frame_id, cam_id),  # image name.
                                    'id': image_cnt + i + 1,  # image number in the entire training set.
                                    'frame_id': frame_id + 1,  # image number in the video sequence, starting from 1.
                                    'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                                    'next_image_id': image_cnt + i + 2 if os.path.exists(os.path.join(data_path, 'images', seq_main, '{}/rgb_:05d_:d.jpg'.format(seq, frame_id+1, cam_id))) else -1,
                                    'video_id': video_cnt,
                                    'height': height,
                                    'width': width}
                        out['images'].append(image_info)

                        ann_p = img_p.replace('images', 'labels').replace('.jpg', '.json')
                        label = json.load(open(ann_p))
                        for track_id, (x_min, y_min, x_max, y_max) in label.items():
                            ann_cnt += 1
                            category_id = 1
                            ann = {'id': ann_cnt,
                                'category_id': category_id,
                                'image_id': image_cnt + i + 1,
                                'track_id': track_id,
                                'bbox': [x_min, y_min, x_max-x_min+1, y_max-y_min+1],
                                'conf': float(1),
                                'iscrowd': 0,
                                'area': float((x_max-x_min+1) * (y_max-y_min+1))}
                            out['annotations'].append(ann)

                        i += 1
                        frame_id+=1
                print('{}: {} images'.format(seq, num_images))

                image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))