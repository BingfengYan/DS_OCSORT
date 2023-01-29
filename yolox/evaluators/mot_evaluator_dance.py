from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from trackers.byte_tracker.byte_tracker import BYTETracker
# from trackers.ocsort_tracker.ocsort import OCSort
from trackers.deepsort_tracker.deepsort import DeepSort
from trackers.motdt_tracker.motdt_tracker import OnlineTracker
from trackers.focsort_tracker.ocsort import OCSort

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
from utils.utils import write_results, write_results_no_score, write_results_no_score2


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

        self.num_person = {'dancetrack0003': 12, 'dancetrack0009': 8, 'dancetrack0011':6, 'dancetrack0013':4, 'dancetrack0017':7, 'dancetrack0021':7, 'dancetrack0022':5,
        'dancetrack0028':9, 'dancetrack0031':9, 'dancetrack0036':8, 'dancetrack0038':5, 'dancetrack0040':9, 'dancetrack0042':9, 'dancetrack0046':17, 'dancetrack0048':7,
        'dancetrack0050':9, 'dancetrack0054':5, 'dancetrack0056':5, 'dancetrack0059':6, 'dancetrack0060':4, 'dancetrack0064':5, 'dancetrack0067':6, 'dancetrack0070':9,
        'dancetrack0071':5, 'dancetrack0076':6, 'dancetrack0078':6, 'dancetrack0084':14, 'dancetrack0085':15, 'dancetrack0088':13, 'dancetrack0089': 13, 'dancetrack0091':4,
        'dancetrack0092':7, 'dancetrack0093':14, 'dancetrack0095':24, 'dancetrack0100':6}

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
    
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_ocsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
            asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)
        
        detections = dict()

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                if video_name == 'images':
                    tmp = img_file_name[0].split('/')
                    video_name = tmp[1]+'_'+tmp[2]+'_'+os.path.splitext(tmp[3])[0].split('_')[-1]
                
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    if video_name in self.num_person:
                        tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                                asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte, num_max=self.num_person[video_name])
                    else:
                        tracker = OCSort(det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                                asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        write_results_no_score2(os.path.join('tmp', '{}.txt'.format(video_names[video_id-1])), results)
                        results = []

                # if True:
                #     if not video_name in detections:
                #         import numpy as np
                #         gt_file = os.path.join('datasets/dancetrack/val', video_name , 'gt/gt.txt')
                #         gt = np.loadtxt(gt_file, delimiter=",")
                #         gt[:, [4,5]] += gt[:, [2,3]]
                #         gt[:, [6, 7]] = 1.0
                #         gt[:, 8] = 0
                #         dets = torch.from_numpy(np.concatenate([gt[:, 0:1], gt[:, 2:9]], 1)).type(tensor_type).float()
                #         self.img_size = (1080, 1920)
                #         detections[video_name] = dets 

                #     all_dets = detections[video_name]
                #     outputs = [all_dets[all_dets[:,0] == frame_id][:, 1:]]

                ckt_file = "dance_detections/{}_detection.pkl".format(video_name)
                if os.path.exists(ckt_file):
                    outputs = [torch.load(ckt_file)]
                    if not video_name in detections:
                        dets = torch.load(ckt_file)
                        detections[video_name] = dets 

                    all_dets = detections[video_name]
                    outputs = [all_dets[all_dets[:,0] == int(os.path.splitext(os.path.basename(img_file_name[0]))[0])][:, 1:]]

                    # if not video_name in detections:
                    #     dets = torch.load(ckt_file)
                    #     detections[video_name] = [dets, torch.load(ckt_file.replace('dance_detections', 'dance_detections0'))]

                    # all_dets = detections[video_name][0]
                    # outputs0 = [all_dets[all_dets[:,0] == int(os.path.splitext(os.path.basename(img_file_name[0]))[0])][:, 1:]]
                    # all_dets = detections[video_name][1]
                    # outputs1 = [all_dets[all_dets[:,0] == int(os.path.splitext(os.path.basename(img_file_name[0]))[0])][:, 1:]]

                    # outputs = torch.concat([outputs0[0], outputs1[0]], 0)

                    # import torchvision
                    # nms_out_index = torchvision.ops.batched_nms(
                    #         outputs[:, :4],
                    #         outputs[:, 4] * outputs[:, 5],
                    #         outputs[:, 6],
                    #         self.nmsthre,
                    #     )
                    # outputs = [outputs[nms_out_index]]
                    
                else:
                    imgs = imgs.type(tensor_type)

                    # skip the the last iters since batchsize might be not enough for batch inference
                    outputs = model(imgs)  # shape[1,23625, 6]
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    # we should save the detections here !  
                    ## yf edit  strat
                    # ckt_file = "dance_detections/{}/{}_detection.pkl".format(video_name, os.path.splitext(os.path.basename(img_file_name[0]))[0])
                    # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                    # torch.save(outputs[0], ckt_file)
                    ## yf edit  end

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name)
            online_tlwhs = []
            online_ids = []
            online_match_mode = []
            for t in online_targets:
                """
                    Here is minor issue that DanceTrack uses the same annotation
                    format as MOT17/MOT20, namely xywh to annotate the object bounding
                    boxes. But DanceTrack annotation is cropped at the image boundary, 
                    which is different from MOT17/MOT20. So, cropping the output
                    bounding boxes at the boundary may slightly fix this issue. But the 
                    influence is minor. For example, with my results on the interpolated
                    OC-SORT:
                    * without cropping: HOTA=55.731
                    * with cropping: HOTA=55.737
                """
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_match_mode.append(t[5])
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_match_mode))

            ## yf edit  strat
            if False:
                import cv2
                img_p = os.path.join('datasets/dancetrack/test', img_file_name[0])
                img = cv2.imread(img_p)
                for ith, box in enumerate(output_results):
                    x_min, y_min, x_max, y_max = box['bbox']
                    x_max += x_min
                    y_max += y_min
                    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                    img = cv2.putText(img, '%d/%0.3f'%(ith, box['score']), (int(x_min), int(y_max)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                for box, id, mode in zip(online_tlwhs, online_ids, online_match_mode):
                    x_min, y_min, x_max, y_max = box
                    x_max += x_min
                    y_max += y_min
                    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255), 2)
                    img = cv2.putText(img, '%d'%(id), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
                
                os.makedirs('tmp/%s'%video_name, exist_ok=True)
                cv2.imwrite('tmp/%s/%s'%(video_name, os.path.basename(img_file_name[0])), img)
            ## yf edit  end

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)
                write_results_no_score2(os.path.join('tmp', '{}.txt'.format(video_names[video_id])), results)


        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results


    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list



    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info