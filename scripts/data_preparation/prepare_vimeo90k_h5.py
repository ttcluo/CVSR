#!/usr/bin/env python3
"""
Vimeo-90K数据集HDF5格式转换脚本
生成事件数据并保存为HDF5格式
"""

import os
import cv2
import h5py
import torch
import numpy as np
import random
import math
from pathlib import Path
from tqdm import tqdm
import esim_py

def voxel_normalization(voxel):
    """
    normalize the voxel same as https://arxiv.org/abs/1912.01584 Section 3.1
    Params:
        voxel: torch.Tensor, shape is [num_bins, H, W]
    return:
        normalized voxel
    """
    # check if voxel all element is 0
    a, b, c = voxel.shape
    tmp = torch.zeros(a, b, c)
    if torch.equal(voxel, tmp):
        return voxel

    abs_voxel, _ = torch.sort(torch.abs(voxel).view(-1, 1).squeeze(1))
    first_non_zero_idx = torch.nonzero(abs_voxel)[0].item()
    non_zero_voxel = abs_voxel[first_non_zero_idx:]
    norm_idx = math.floor(non_zero_voxel.shape[0] * 0.98)

    ones = torch.ones_like(voxel)
    normed_voxel = torch.where(torch.abs(voxel) < non_zero_voxel[norm_idx],
                              voxel / non_zero_voxel[norm_idx], voxel)
    normed_voxel = torch.where(normed_voxel >= non_zero_voxel[norm_idx], ones, normed_voxel)
    normed_voxel = torch.where(normed_voxel <= -non_zero_voxel[norm_idx], -ones, normed_voxel)
    return normed_voxel

def events_to_voxel_torch(xs, ys, ts, ps, bins, sensor_size=None):
    """
    Convert events to voxel grid representation
    """
    if sensor_size is None:
        sensor_size = (xs.max().int().item() + 1, ys.max().int().item() + 1)

    # Normalize timestamps
    if len(ts) == 0:
        return torch.zeros((bins, sensor_size[1], sensor_size[0]))

    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
    ts_norm = ts_norm * (bins - 1)

    # Create voxel grid
    voxel = torch.zeros((bins, sensor_size[1], sensor_size[0]))

    for i in range(len(xs)):
        x, y = int(xs[i]), int(ys[i])
        t_idx = int(ts_norm[i])
        pol = ps[i]

        if 0 <= x < sensor_size[0] and 0 <= y < sensor_size[1] and 0 <= t_idx < bins:
            voxel[t_idx, y, x] += pol

    return voxel

def create_timestamps_file(num_frames, fps=25.0):
    """Create timestamps for frames"""
    timestamps = []
    for i in range(num_frames):
        timestamps.append(i / fps)
    return timestamps

def simulate_events_from_frames(frame_paths, height, width, fps=25.0):
    """
    Simulate events from frame sequence using esim_py
    """
    # Configuration for event simulation
    config = {
        'refractory_period': 1e-4,
        'CT_range': [0.05, 0.5],
        'max_CT': 0.5,
        'min_CT': 0.02,
        'mu': 1,
        'sigma': 0.1,
        'H': height,
        'W': width,
        'log_eps': 1e-3,
        'use_log': True,
    }

    # Random contrast thresholds
    Cp = random.uniform(config['CT_range'][0], config['CT_range'][1])
    Cn = random.gauss(config['mu'], config['sigma']) * Cp
    Cp = min(max(Cp, config['min_CT']), config['max_CT'])
    Cn = min(max(Cn, config['min_CT']), config['max_CT'])

    # Create event simulator
    esim = esim_py.EventSimulator(Cp, Cn, config['refractory_period'],
                                 config['log_eps'], config['use_log'])

    # Create temporary directory for frames and timestamps
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)

    # Copy frames to temp directory and create timestamps
    timestamps = create_timestamps_file(len(frame_paths), fps)
    timestamps_file = temp_dir / "timestamps.txt"

    with open(timestamps_file, 'w') as f:
        for i, (frame_path, ts) in enumerate(zip(frame_paths, timestamps)):
            # Copy frame to temp directory
            frame = cv2.imread(str(frame_path))
            temp_frame_path = temp_dir / f"{i:06d}.png"
            cv2.imwrite(str(temp_frame_path), frame)
            f.write(f"{ts}\n")

    # Generate events
    try:
        events = esim.generateFromFolder(str(temp_dir), str(timestamps_file))
    except Exception as e:
        print(f"Warning: Event simulation failed: {e}")
        # Return empty events if simulation fails
        events = np.zeros((0, 4))

    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)

    return events

def process_vimeo_sequence(sequence_path, output_path, is_hr=True):
    """
    Process a single Vimeo sequence and save as HDF5
    """
    # Get all frame files
    frame_files = sorted([f for f in sequence_path.glob("*.png")])
    if len(frame_files) == 0:
        frame_files = sorted([f for f in sequence_path.glob("*.jpg")])

    if len(frame_files) == 0:
        print(f"No frames found in {sequence_path}")
        return False

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"Cannot read frame {frame_files[0]}")
        return False

    height, width = first_frame.shape[:2]

    # Simulate events from frames
    print(f"Simulating events for {sequence_path.name}...")
    events = simulate_events_from_frames(frame_files, height, width)

    # Create HDF5 file
    os.makedirs(output_path.parent, exist_ok=True)

    with h5py.File(output_path, 'w') as h5f:
        # Create groups
        images_group = h5f.create_group('images')
        voxels_f_group = h5f.create_group('voxels_f')
        voxels_b_group = h5f.create_group('voxels_b')

        # Process each frame
        for i, frame_file in enumerate(frame_files):
            # Read and store image
            frame = cv2.imread(str(frame_file))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images_group.create_dataset(f'{i:06d}', data=frame_rgb)

            # Generate voxel grids for this frame
            if len(events) > 0:
                # Convert events to torch tensors
                xs = torch.from_numpy(events[:, 0]).float()
                ys = torch.from_numpy(events[:, 1]).float()
                ts = torch.from_numpy(events[:, 2]).float()
                ps = torch.from_numpy(events[:, 3]).float()

                # Filter events for current frame time window
                frame_start_time = i / 25.0
                frame_end_time = (i + 1) / 25.0

                mask = (ts >= frame_start_time) & (ts < frame_end_time)
                if mask.sum() > 0:
                    xs_frame = xs[mask]
                    ys_frame = ys[mask]
                    ts_frame = ts[mask]
                    ps_frame = ps[mask]

                    # Forward voxel
                    voxel_f = events_to_voxel_torch(xs_frame, ys_frame, ts_frame, ps_frame,
                                                   bins=5, sensor_size=(width, height))
                    voxel_f = voxel_normalization(voxel_f)
                    voxels_f_group.create_dataset(f'{i:06d}', data=voxel_f.numpy())

                    # Backward voxel
                    xs_b = torch.flip(xs_frame, dims=[0])
                    ys_b = torch.flip(ys_frame, dims=[0])
                    ts_b = torch.flip(frame_end_time - ts_frame + frame_start_time, dims=[0])
                    ps_b = torch.flip(-ps_frame, dims=[0])

                    voxel_b = events_to_voxel_torch(xs_b, ys_b, ts_b, ps_b,
                                                   bins=5, sensor_size=(width, height))
                    voxel_b = voxel_normalization(voxel_b)
                    voxels_b_group.create_dataset(f'{i:06d}', data=voxel_b.numpy())
                else:
                    # Empty voxel grids if no events
                    empty_voxel = np.zeros((5, height, width))
                    voxels_f_group.create_dataset(f'{i:06d}', data=empty_voxel)
                    voxels_b_group.create_dataset(f'{i:06d}', data=empty_voxel)
            else:
                # Empty voxel grids if no events
                empty_voxel = np.zeros((5, height, width))
                voxels_f_group.create_dataset(f'{i:06d}', data=empty_voxel)
                voxels_b_group.create_dataset(f'{i:06d}', data=empty_voxel)

    print(f"Successfully created {output_path}")
    return True

def main():
    # 设置路径
    vimeo_root = Path("/data/luochuan/vsr_dataset/vimeo90k_small")
    hr_path = vimeo_root / "GT"
    lr_path = vimeo_root / "BIx4"

    output_root = Path("Vimeo_h5")

    if not hr_path.exists():
        print(f"Error: HR path {hr_path} does not exist!")
        return

    if not lr_path.exists():
        print(f"Error: LR path {lr_path} does not exist!")
        return

    # 处理训练集和测试集
    for split in ['train', 'test']:
        hr_split_path = hr_path / split
        lr_split_path = lr_path / split

        if not hr_split_path.exists():
            print(f"Skipping {split} - HR path does not exist")
            continue

        if not lr_split_path.exists():
            print(f"Skipping {split} - LR path does not exist")
            continue

        # Get all sequences
        hr_sequences = []
        for seq_dir in hr_split_path.iterdir():
            if seq_dir.is_dir():
                hr_sequences.append(seq_dir)

        hr_sequences.sort()

        print(f"\nProcessing {split} set: {len(hr_sequences)} sequences")

        # Process each sequence
        for seq_dir in tqdm(hr_sequences, desc=f"Processing {split}"):
            seq_name = seq_dir.name

            # HR processing
            hr_output_path = output_root / "HR" / split / f"{seq_name}.h5"
            if not hr_output_path.exists():
                success = process_vimeo_sequence(seq_dir, hr_output_path, is_hr=True)
                if not success:
                    print(f"Failed to process HR sequence {seq_name}")

            # LR processing
            lr_seq_path = lr_split_path / seq_name
            if lr_seq_path.exists():
                lr_output_path = output_root / "LRx4" / split / f"{seq_name}.h5"
                if not lr_output_path.exists():
                    success = process_vimeo_sequence(lr_seq_path, lr_output_path, is_hr=False)
                    if not success:
                        print(f"Failed to process LR sequence {seq_name}")
            else:
                print(f"Warning: LR sequence {seq_name} not found")

    print("\nDataset conversion completed!")

if __name__ == "__main__":
    main()

