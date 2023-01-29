import os
import numpy as np
import torch

root = 'dance_detections'
dance_names = os.listdir(root)

for dance_name in dance_names:
    if not os.path.isdir(os.path.join(root, dance_name)): continue
    frame_names = os.listdir(os.path.join(root, dance_name))
    if os.path.exists(os.path.join(root, '{}_detection.pkl'.format(dance_name))): continue
    outputs = []
    for frame in frame_names:
        output = torch.load(os.path.join(root, dance_name, frame))
        
        frame_id = torch.ones([len(output), 1], dtype=output.dtype, device=output.device) * int(frame.split('_')[0])
        outputs.append(torch.cat((frame_id, output), 1))

    torch.save(torch.vstack(outputs), os.path.join(root, '{}_detection.pkl'.format(dance_name)))