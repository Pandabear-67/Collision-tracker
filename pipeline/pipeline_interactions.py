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

# --- RAFT PATH SETUP ---
raft_path = '/content/drive/MyDrive/accident_files/RAFT'
for path in [raft_path, f'{raft_path}/core']:
    if path not in sys.path:
        sys.path.append(path)

from core.raft import RAFT
from utils.utils import InputPadder

# --- 1. Data Loaders & Models ---

class FrameLoader:
    def __init__(self, frame_dir):
        self.frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    def load(self):
        for f in self.frames:
            yield f, cv2.imread(f)

class DetectorTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def update(self, frame):
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.35, verbose=False
        )[0]

        boxes = results.boxes
        if boxes is None or boxes.id is None:
            return [], []

        return (
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int)
        )

class OpticalFlow:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        args = Namespace(small=False, mixed_precision=False)
        self.model = RAFT(args)

        state_dict = torch.load(ckpt_path)
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k.replace("module.", "")] = v

        self.model.load_state_dict(new_state)
        self.model.to(device).eval()

    @torch.no_grad()
    def compute_flow(self, img1, img2):
        img1 = torch.from_numpy(img1).permute(2,0,1).float().unsqueeze(0).to(self.device)
        img2 = torch.from_numpy(img2).permute(2,0,1).float().unsqueeze(0).to(self.device)

        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        _, flow = self.model(img1, img2, iters=12, test_mode=True)
        return flow[0]

# --- IoU helper ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# --- 2. Feature Extraction ---

class FeatureBuilder:
    def __init__(self, smoothing_factor=0.3):
        self.alpha = smoothing_factor
        self.prev_centroids = {}
        self.prev_vel = {}
        self.smoothed_coords = {}
        self.prev_iou = {}

    def bbox_to_features(self, boxes, ids, frame_shape):
        H, W = frame_shape[:2]
        n = len(boxes)

        if n == 0:
            return np.empty((0, 19), dtype=np.float32)

        base_feats = []
        centroids = []
        velocities = []

        # ---- PASS 1: per-object ----
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = box

            cx_raw = (x1 + x2) / 2 / W
            cy_raw = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            if tid in self.smoothed_coords:
                pcx, pcy = self.smoothed_coords[tid]
                cx = self.alpha * cx_raw + (1 - self.alpha) * pcx
                cy = self.alpha * cy_raw + (1 - self.alpha) * pcy
            else:
                cx, cy = cx_raw, cy_raw

            self.smoothed_coords[tid] = (cx, cy)

            vx = vy = speed = ax = ay = 0.0

            if tid in self.prev_centroids:
                px, py = self.prev_centroids[tid]
                vx_raw, vy_raw = cx - px, cy - py

                if tid in self.prev_vel:
                    pvx, pvy = self.prev_vel[tid]
                    vx = self.alpha * vx_raw + (1 - self.alpha) * pvx
                    vy = self.alpha * vy_raw + (1 - self.alpha) * pvy
                    ax, ay = vx - pvx, vy - pvy
                else:
                    vx, vy = vx_raw, vy_raw

                speed = np.sqrt(vx**2 + vy**2)

            self.prev_centroids[tid] = (cx, cy)
            self.prev_vel[tid] = (vx, vy)

            base_feats.append([cx, cy, bw, bh, vx, vy, speed, ax, ay])
            centroids.append((cx, cy))
            velocities.append((vx, vy))

        base_feats = np.array(base_feats)
        centroids = np.array(centroids)
        velocities = np.array(velocities)

        # ---- PASS 2: interaction ----
        interaction_feats = []

        for i in range(n):
            cx_i, cy_i = centroids[i]
            vx_i, vy_i = velocities[i]

            min_dist = 1e6
            best_closing = 0.0
            best_ttc = 0.0
            best_iou = 0.0
            best_angle = 0.0
            best_pair = None

            for j in range(n):
                if i == j:
                    continue

                cx_j, cy_j = centroids[j]
                vx_j, vy_j = velocities[j]

                dx, dy = cx_j - cx_i, cy_j - cy_i
                dist = np.sqrt(dx**2 + dy**2) + 1e-6

                dvx, dvy = vx_j - vx_i, vy_j - vy_i
                closing = (dx * dvx + dy * dvy) / dist

                if dist < min_dist:
                    min_dist = dist
                    best_closing = closing
                    best_pair = (ids[i], ids[j])

                    # TTC
                    if closing < 0:
                        best_ttc = dist / (-closing + 1e-6)
                    else:
                        best_ttc = 0.0

                    # IoU
                    best_iou = compute_iou(boxes[i], boxes[j])

                    # angle
                    v_norm = np.sqrt(vx_i**2 + vy_i**2) + 1e-6
                    cos_theta = (vx_i * dx + vy_i * dy) / (v_norm * dist)
                    best_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

            # IoU delta
            delta_iou = 0.0
            if best_pair is not None:
                prev = self.prev_iou.get(best_pair, 0.0)
                delta_iou = best_iou - prev
                self.prev_iou[best_pair] = best_iou

            interaction_feats.append([
                min_dist,
                best_closing,
                best_ttc,
                best_iou,
                delta_iou,
                best_angle
            ])

        interaction_feats = np.array(interaction_feats)

        return np.concatenate([base_feats, interaction_feats], axis=1)

# --- Flow pooling (unchanged) ---
def pool_flow(flow, boxes):
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    flow = flow.permute(1,2,0)
    out = []

    for x1,y1,x2,y2 in boxes:
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        patch = flow[y1:y2, x1:x2]

        if patch.numel() == 0:
            out.append([0,0,0,0])
            continue

        mean = patch.mean(dim=(0,1))
        mag = torch.norm(patch, dim=-1)

        out.append([
            mean[0].item(),
            mean[1].item(),
            mag.mean().item(),
            mag.std().item()
        ])

    return np.array(out, dtype=np.float32)

# --- 3. Sequence + I/O ---

class SequenceBuilder:
    def __init__(self, seq_len=16, max_objects=50, feat_dim=19):
        self.buffer = deque(maxlen=seq_len)
        self.seq_len = seq_len
        self.max_objects = max_objects
        self.feat_dim = feat_dim

    def add_frame(self, frame_feats):
        if frame_feats is None or len(frame_feats) == 0:
            frame_feats = np.zeros((1, self.feat_dim), dtype=np.float32)

        self.buffer.append(frame_feats)

        if len(self.buffer) < self.seq_len:
            return None

        seq = []
        for f in self.buffer:
            pad = np.zeros((self.max_objects, self.feat_dim), dtype=np.float32)
            n = min(len(f), self.max_objects)
            pad[:n] = f[:n]
            seq.append(pad)

        return np.stack(seq)

class HDF5Writer:
    def __init__(self, path, feat_dim=19):
        self.h5 = h5py.File(path, "w")
        self.X = self.h5.create_dataset(
            "X",
            shape=(0,16,50,feat_dim),
            maxshape=(None,16,50,feat_dim),
            chunks=True,
            dtype="float32"
        )
        self.ptr = 0

    def write(self, seq):
        if seq is not None:
            self.X.resize(self.ptr+1, axis=0)
            self.X[self.ptr] = seq
            self.ptr += 1

    def close(self):
        self.h5.close()