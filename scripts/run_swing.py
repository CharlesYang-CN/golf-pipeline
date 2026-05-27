"""Wrapper for Swing-Phase-Detection inference.

Usage: conda run -n env_swing python scripts/run_swing.py <video_path> <output_json>
Outputs JSON: {fps, total_frames, events: [{event, frame, time_s, confidence}]}
"""
import argparse
import json
import sys
import os

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from ultralytics import YOLO

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


class ToTensor:
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class PreprocessedVideo:
    def __init__(self, path, dim=160, transform=None):
        self.transform = transform
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        x1, y1, x2, y2 = _get_person_bbox(path)

        images = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret:
                break
            crop = frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]
            ratio = dim / max(crop_h, crop_w)
            new_size = (int(crop_w * ratio), int(crop_h * ratio))
            resized = cv2.resize(crop, new_size)
            delta_w = dim - new_size[0]
            delta_h = dim - new_size[1]
            left, right = delta_w // 2, delta_w - delta_w // 2
            top, bottom = delta_h // 2, delta_h - delta_h // 2
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT,
                                        value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            images.append(padded_rgb)
        cap.release()
        self.images = np.asarray(images)
        self.labels = np.zeros(len(images))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = {'images': self.images, 'labels': self.labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


def _get_person_bbox(video_path, margin=0.2, num_samples=10):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    yolo = YOLO('yolo11n.pt')

    boxes = []
    for frame_idx in np.linspace(0, total - 1, min(num_samples, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        results = yolo(frame, verbose=False)
        for b in results[0].boxes:
            if int(b.cls[0]) == 0:
                boxes.append(b.xyxy[0].cpu().numpy())
    cap.release()

    if not boxes:
        raise RuntimeError('No person detected in video')

    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes.mean(axis=0).astype(int)
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)


def run_inference(video_path, output_path, seq_length=64):
    original_dir = os.getcwd()
    swing_dir = '/app/swing'
    os.chdir(swing_dir)
    sys.path.insert(0, swing_dir)

    from model import EventDetector

    transform = transforms.Compose([ToTensor(),
                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = PreprocessedVideo(video_path, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    fps = ds.fps

    model = EventDetector(pretrain=True, width_mult=1., lstm_layers=1, lstm_hidden=256,
                          bidirectional=True, dropout=False)
    save_dict = torch.load(os.path.join(swing_dir, 'models', 'swingnet_2000.pth.tar'))
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()

    for sample in dl:
        images = sample['images']
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    confidences = [float(probs[e, i]) for i, e in enumerate(events)]

    os.chdir(original_dir)

    results = {
        'fps': float(fps),
        'total_frames': int(images.shape[1]),
        'events': []
    }
    for i, (e, c) in enumerate(zip(events, confidences)):
        t = e / fps
        results['events'].append({
            'event': event_names[i],
            'frame': int(e),
            'time_s': round(t, 3),
            'confidence': round(c, 3)
        })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Swing detection done: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Path to input video')
    parser.add_argument('output', help='Path to output JSON')
    parser.add_argument('-s', '--seq-length', type=int, default=64)
    args = parser.parse_args()
    run_inference(args.video, args.output, args.seq_length)
