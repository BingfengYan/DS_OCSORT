# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


from collections import defaultdict
from glob import glob
import json
import os
import cv2
import subprocess
from tqdm import tqdm


def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]

# with open("/data/Dataset/mot/det_db_oc_sort.json") as f:
#     det_db = json.load(f)

def process(trk_path, img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape
    command = [
        "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/yanfeng/software/anaconda3/envs/detrex/bin/ffmpeg",
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{w}x{h}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', '20',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-s', f'{w//2*2}x{h//2*2}',
        '-an',  # Tells FFMPEG not to expect any audio
        '-loglevel', 'error',
        # '-crf', '26',
        '-b:v', '0',
        '-pix_fmt', 'yuv420p'
    ]
    # writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)
    fps = 16 
    size = (w,h) 
    videowriter = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)


    tracklets = defaultdict(list)
    for line in open(trk_path):
        t, id, *xywhs = line.split(',')[:7]
        t, id = int(t), int(float(id))
        x, y, w, h, s = map(float, xywhs)
        tracklets[t].append((int(id), *map(int, (x, y, x+w, y+h))))

    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        # for det in det_db[path.replace('.jpg', '.txt')]:
        #     x1, y1, w, h, _ = map(int, map(float, det.strip().split(',')))
        #     im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 6)
        for j, x1, y1, x2, y2 in tracklets[i + 1]:
            im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 4)
            im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, get_color(j), 2)
        # writing_process.stdin.write(im.tobytes())
        videowriter.write(im)
        
    videowriter.release()


if __name__ == '__main__':
    jobs = os.listdir("YOLOX_outputs/tmp/tmp_test/")
    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    jobs = sorted(jobs)[rank::ws]
    for seq in jobs:
        seq = 'dancetrack0085.txt'
        print(seq)
        trk_path = "YOLOX_outputs/tmp/tmp_test/" + seq
        # trk_path = "/data/Dataset/mot/DancdancetrackeTrack/val/dancetrack0010/gt/gt.txt"

        img_list = glob(f"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-vacv/yanfeng/data/dancetrack/test/{seq[:-4]}/img1/*.jpg")
        process(trk_path, img_list, f'tmp/{seq[:-4]}.avi')
        break
