#imports
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import sys
import os
from argparse import Namespace
import numpy as np
import h5py
from collections import defaultdict, deque
import cv2
import glob
from argparse import Namespace
from collections import OrderedDict 

#Class to load frames
class FrameLoader:
    def __init__(self, frame_dir):
        self.frame_dir = frame_dir
        self.frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    def parse_metadata(self, filename):
        # Town03_head-on_clear_22_frame_0000.jpg
        base = os.path.basename(filename).replace(".jpg", "")
        parts = base.split("_")

        scene_id = "_".join(parts[:-2])
        frame_idx = int(parts[-1])
        return scene_id, frame_idx

    def load(self):
        for f in self.frames:
            img = cv2.imread(f)
            yield f, img

#Detector and Tracker using YOLOv8 with ByteTrack
class DetectorTracker:
    def __init__(self, model_path="/content/drive/MyDrive/accident_files/yolo26n_apr5.pt"):
        self.model = YOLO(model_path)

    def update(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,  #new addition
            iou=0.5, #new additions
            verbose=False
        )[0]

        boxes = results.boxes

        # if boxes is None:
        #     return []
            
        if boxes is None or boxes.id is None:
            return [], []

        xyxy = boxes.xyxy.cpu().numpy()
        # ids = boxes.id.cpu().numpy() if boxes.id is not None else [-1]*len(xyxy)
        ids = boxes.id.cpu().numpy().astype(int) 
        
        return xyxy, ids

# Add RAFT and core directories to the system path
# This allows 'core.raft' to be found and also lets 'raft.py' find its local dependencies like 'update'
raft_path = '/content/drive/MyDrive/accident_files/RAFT'
for path in [raft_path, f'{raft_path}/core']:
    if path not in sys.path:
        sys.path.append(path)

from core.raft import RAFT
from utils.utils import InputPadder

# RAFT-based optical flow computation
class OpticalFlow:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        self.model = RAFT(args)

        # Load state dict and handle the 'module.' prefix from DataParallel
        state_dict = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def compute_flow(self, img1, img2):
        img1 = torch.from_numpy(img1).permute(2,0,1).float().unsqueeze(0).to(self.device)
        img2 = torch.from_numpy(img2).permute(2,0,1).float().unsqueeze(0).to(self.device)

        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        flow_low, flow_up = self.model(img1, img2, iters=12, test_mode=True)

        return flow_up[0]

# Feature builder to convert bounding boxes and flow into feature vectors
class FeatureBuilder:
    # def __init__(self):
    def __init__(self, smoothing_factor = 0.3):
        self.prev_centroids = {}
        self.prev_vel = {}

        #new code
        self.alpha = smoothing_factor
        self.smoothed_features = {}
        
    def bbox_to_features(self, boxes, ids, flow, frame_shape):
        # H, W = frame_shape[:2]

        # feats = []

        # for box, tid in zip(boxes, ids):
        #     x1, y1, x2, y2 = box

        #     cx = (x1 + x2) / 2 / W
        #     cy = (y1 + y2) / 2 / H
        #     bw = (x2 - x1) / W
        #     bh = (y2 - y1) / H

        #     # velocity
        #     if tid in self.prev_centroids:
        #         px, py = self.prev_centroids[tid]
        #         vx = cx - px
        #         vy = cy - py
        #     else:
        #         vx, vy = 0.0, 0.0

        #     speed = (vx**2 + vy**2) ** 0.5

        #     # acceleration
        #     if tid in self.prev_vel:
        #         pvx, pvy = self.prev_vel[tid]
        #         ax = vx - pvx
        #         ay = vy - pvy
        #     else:
        #         ax, ay = 0.0, 0.0

        #     self.prev_centroids[tid] = (cx, cy)
        #     self.prev_vel[tid] = (vx, vy)

        #     feats.append([cx, cy, bw, bh, vx, vy, speed, ax, ay])

        # return np.array(feats, dtype=np.float32)
        
        H, W = frame_shape[:2]
        feats = []

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = box

            # 1. Normalize Coordinates
            cx_raw = (x1 + x2) / 2 / W
            cy_raw = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            # 2. Apply EMA Smoothing to Centroids
            # This prevents the "vibration" of the bounding box from creating fake speed
            if tid in self.smoothed_features:
                prev_cx, prev_cy = self.smoothed_features[tid]
                cx = (self.alpha * cx_raw) + ((1 - self.alpha) * prev_cx)
                cy = (self.alpha * cy_raw) + ((1 - self.alpha) * prev_cy)
            else:
                cx, cy = cx_raw, cy_raw
            
            self.smoothed_features[tid] = (cx, cy)

            # 3. Velocity Calculation with Smoothing
            if tid in self.prev_centroids:
                px, py = self.prev_centroids[tid]
                vx_raw = cx - px
                vy_raw = cy - py
                
                # Smooth the velocity to prevent acceleration spikes
                if tid in self.prev_vel:
                    pvx, pvy = self.prev_vel[tid]
                    vx = (self.alpha * vx_raw) + ((1 - self.alpha) * pvx)
                    vy = (self.alpha * vy_raw) + ((1 - self.alpha) * pvy)
                else:
                    vx, vy = vx_raw, vy_raw
            else:
                vx, vy = 0.0, 0.0

            speed = (vx**2 + vy**2) ** 0.5

            # 4. Acceleration Calculation
            if tid in self.prev_vel:
                pvx, pvy = self.prev_vel[tid]
                ax = vx - pvx
                ay = vy - pvy
            else:
                ax, ay = 0.0, 0.0

            # Update history for next frame
            self.prev_centroids[tid] = (cx, cy)
            self.prev_vel[tid] = (vx, vy)

            feats.append([cx, cy, bw, bh, vx, vy, speed, ax, ay])

        # Clean up tracking history for lost IDs to save memory
        current_ids = set(ids)
        for lost_id in list(self.smoothed_features.keys()):
            if lost_id not in current_ids:
                del self.smoothed_features[lost_id]
                if lost_id in self.prev_centroids: del self.prev_centroids[lost_id]
                if lost_id in self.prev_vel: del self.prev_vel[lost_id]

        return np.array(feats, dtype=np.float32)

# Pool optical flow within each bounding box to get motion features
def pool_flow(flow, boxes, W, H):
    # flow: (2, H, W)
    # boxes: Nx4 in pixel coords

    flow = flow.permute(1,2,0)  # H,W,2

    out = []

    for x1,y1,x2,y2 in boxes:
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])

        patch = flow[y1:y2, x1:x2]

        mean = patch.mean(dim=(0,1))
        std = patch.std(dim=(0,1))

        mag = torch.norm(patch, dim=-1)

        out.append([
            mean[0].item(),
            mean[1].item(),
            mag.mean().item(),
            mag.std().item()
        ])

    return np.array(out, dtype=np.float32)

args = Namespace(small=False, mixed_precision=False)

SEQ_LEN = 16

# Sequence builder to maintain a buffer of recent frames and build sequences of features
class SequenceBuilder:
    def __init__(self, max_objects=50, feat_dim=13):
        self.buffer = deque(maxlen=SEQ_LEN)
        self.max_objects = max_objects
        self.feat_dim = feat_dim

    def add_frame(self, frame_feats):
        """
        frame_feats: (num_objects, feat_dim)
        """
        self.buffer.append(frame_feats)

        if len(self.buffer) < SEQ_LEN:
            return None

        return self.build_sequence()

    def build_sequence(self):
        seq = []

        for f in self.buffer:
            pad = np.zeros((self.max_objects, self.feat_dim), dtype=np.float32)
            n = min(len(f), self.max_objects)
            pad[:n] = f[:n]
            seq.append(pad)

        return np.stack(seq)  # (16, max_objects, feat_dim)

# HDF5 writer to save sequences to file
class HDF5Writer:
    def __init__(self, path):
        self.h5 = h5py.File(path, "w")
        self.X = self.h5.create_dataset(
            "X",
            shape=(0, 16, 50, 13),
            maxshape=(None, 16, 50, 13),
            chunks=True,
            dtype="float32"
        )
        self.ptr = 0

    def write(self, seq):
        self.X.resize(self.ptr + 1, axis=0)
        self.X[self.ptr] = seq
        self.ptr += 1

    def close(self):
        self.h5.close()