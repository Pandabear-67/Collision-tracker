import os
import glob
from ultralytics import YOLO
import torch
import h5py
import numpy as np
from collections import deque, OrderedDict
from argparse import Namespace
import cv2
import sys

# Ensure RAFT is in your path
raft_path = '/content/drive/MyDrive/accident_files/RAFT'
for path in [raft_path, f'{raft_path}/core']:
    if path not in sys.path:
        sys.path.append(path)

from core.raft import RAFT
from utils.utils import InputPadder
from ultralytics import YOLO

# --- 1. Data Loaders & Models ---

class FrameLoader:
    def __init__(self, frame_dir):
        self.frame_dir = frame_dir
        self.frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    def load(self):
        for f in self.frames:
            img = cv2.imread(f)
            yield f, img

class DetectorTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def update(self, frame):
        # conf=0.35 reduces low-confidence flickering boxes
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=0.35, verbose=False
        )[0]
        boxes = results.boxes
        if boxes is None or boxes.id is None:
            return [], []
        
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        return xyxy, ids

class OpticalFlow:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        args = Namespace(small=False, mixed_precision=False)
        self.model = RAFT(args)

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
        _, flow_up = self.model(img1, img2, iters=12, test_mode=True)
        return flow_up[0]

# --- 2. Feature Extraction & Smoothing ---

class FeatureBuilder:
    def __init__(self, smoothing_factor=0.3):
        self.prev_centroids = {}
        self.prev_vel = {}
        self.alpha = smoothing_factor
        self.smoothed_coords = {} 

    def bbox_to_features(self, boxes, ids, frame_shape):
        H, W = frame_shape[:2]
        if len(boxes) == 0:
            return np.empty((0, 9), dtype=np.float32)

        feats = []
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = box
            
            # Coordinate Normalization
            cx_raw, cy_raw = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H

            # Apply EMA Smoothing to Centroids
            if tid in self.smoothed_coords:
                prev_cx, prev_cy = self.smoothed_coords[tid]
                cx = (self.alpha * cx_raw) + ((1 - self.alpha) * prev_cx)
                cy = (self.alpha * cy_raw) + ((1 - self.alpha) * prev_cy)
            else:
                cx, cy = cx_raw, cy_raw
            
            self.smoothed_coords[tid] = (cx, cy)

            # Velocity & Acceleration calculation
            vx, vy, speed, ax, ay = 0.0, 0.0, 0.0, 0.0, 0.0
            if tid in self.prev_centroids:
                px, py = self.prev_centroids[tid]
                vx_raw, vy_raw = cx - px, cy - py
                
                # Smooth the velocity vector
                if tid in self.prev_vel:
                    pvx, pvy = self.prev_vel[tid]
                    vx = (self.alpha * vx_raw) + ((1 - self.alpha) * pvx)
                    vy = (self.alpha * vy_raw) + ((1 - self.alpha) * pvy)
                    ax, ay = vx - pvx, vy - pvy
                else:
                    vx, vy = vx_raw, vy_raw

                speed = (vx**2 + vy**2) ** 0.5

            self.prev_centroids[tid] = (cx, cy)
            self.prev_vel[tid] = (vx, vy)
            feats.append([cx, cy, bw, bh, vx, vy, speed, ax, ay])

        return np.array(feats, dtype=np.float32)

def pool_flow(flow, boxes):
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    
    flow_perm = flow.permute(1, 2, 0)
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        patch = flow_perm[y1:y2, x1:x2]
        
        if patch.numel() == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue

        mean = patch.mean(dim=(0, 1))
        mag = torch.norm(patch, dim=-1)
        out.append([mean[0].item(), mean[1].item(), mag.mean().item(), mag.std().item()])
    
    return np.array(out, dtype=np.float32)

# --- 3. Sequence Building & I/O ---

class SequenceBuilder:
    def __init__(self, seq_len=16, max_objects=50, feat_dim=13):
        self.buffer = deque(maxlen=seq_len)
        self.seq_len = seq_len
        self.max_objects = max_objects
        self.feat_dim = feat_dim

    def add_frame(self, frame_feats):
        # Handle empty frames
        if frame_feats is None or len(frame_feats) == 0:
            frame_feats = np.zeros((1, self.feat_dim), dtype=np.float32)
            
        self.buffer.append(frame_feats)
        if len(self.buffer) < self.seq_len:
            return None

        # Pad to (16, 50, 13)
        seq = []
        for f in self.buffer:
            pad = np.zeros((self.max_objects, self.feat_dim), dtype=np.float32)
            n = min(len(f), self.max_objects)
            pad[:n] = f[:n]
            seq.append(pad)
        return np.stack(seq)

class HDF5Writer:
    def __init__(self, path, feat_dim=13):
        self.h5 = h5py.File(path, "w")
        self.feat_dim = feat_dim
        # Shape: (Samples, Timesteps, Max_Objects, Features)
        self.X = self.h5.create_dataset(
            "X",
            shape=(0, 16, 50, self.feat_dim),
            maxshape=(None, 16, 50, self.feat_dim),
            chunks=True,
            dtype="float32"
        )
        self.ptr = 0

    def write(self, seq):
        if seq is not None:
            self.X.resize(self.ptr + 1, axis=0)
            self.X[self.ptr] = seq
            self.ptr += 1

    def close(self):
        self.h5.close()