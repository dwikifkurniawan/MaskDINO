# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
#
# This script is adapted from train_net.py for pure FPS benchmarking
# using the accurate synchronization method.
# ------------------------------------------------------------------------
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    # Corrected the typo in the warning type
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import time
import logging
import os
import torch
import itertools

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskDINO imports to register modules
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)

# A custom Trainer class that we can borrow methods from,
# simplified from train_net.py
class Trainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # This is the default D2 test loader, which is what we want.
        # It will use the TEST config, including INPUT.MIN_SIZE_TEST.
        return super().build_test_loader(cfg, dataset_name)

def setup(args):
    """
    Create configs and perform basic setups.
    Copied from train_net.py
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg

@torch.no_grad()
def benchmark_fps(model, data_loader, warmup_runs, num_runs):
    """
    Runs the FPS benchmark using the logic from your script, iboy.
    Includes warm-up and torch.cuda.synchronize() for accuracy.
    """
    model.eval()
    device = next(model.parameters()).device
    timings = []
    
    data_iter = iter(data_loader)
    total_images = 0

    # --- Warm-up Phase ---
    print(f"Performing {warmup_runs} warm-up runs...")
    for i in range(warmup_runs):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)
        
        # Move data to GPU (model expects a list of dicts)
        data = [{"image": d["image"].to(device), "height": d["height"], "width": d["width"]} for d in data]
        _ = model(data)
    
    print("Warm-up complete. Starting benchmark...")

    # --- Benchmark Phase ---
    for i in range(num_runs):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)

        # Move data to GPU (model expects a list of dicts)
        data = [{"image": d["image"].to(device), "height": d["height"], "width": d["width"]} for d in data]
        batch_size = len(data)

        # Synchronize before starting the timer
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        
        # The actual model inference
        _ = model(data)
        
        # Synchronize again to wait for the inference to complete
        torch.cuda.synchronize(device)
        end_time = time.perf_counter()
        
        timings.append(end_time - start_time)
        total_images += batch_size

    # --- Results Calculation ---
    if not timings:
        print("Error: No images were processed. Cannot calculate FPS.")
        return

    total_time = sum(timings)
    fps = total_images / total_time

    print("\n" + "="*40)
    print("--- Performance Test Results ---")
    print(f"Config: {args.config_file}")
    print(f"Model: {cfg.MODEL.WEIGHTS}")
    print(f"Input Size (Short Edge): {cfg.INPUT.MIN_SIZE_TEST}")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Total batches tested: {num_runs}")
    print(f"Total images tested: {total_images}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print(f"Avg. Latency per Batch: {total_time / num_runs * 1000:.2f} ms")
    print("="*40)


def main(args):
    cfg = setup(args)
    
    # --- 1. Build Model ---
    model = Trainer.build_model(cfg)
    
    # --- 2. Load Weights ---
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    model.eval()

    # --- 3. Build Data Loader ---
    # We use the first dataset from the TEST config
    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    
    # --- 4. Run Benchmark ---
    benchmark_fps(model, data_loader, args.warmup_runs, args.num_runs)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--warmup-runs", type=int, default=20, help="Number of initial runs to discard for GPU warm-up."
    )
    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of batches to run for the benchmark."
    )
    args = parser.parse_args()
    
    print("Command Line Args:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

