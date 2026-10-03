import os
import json
import yaml
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from dataset import LandslideDataset
from SAM2UNet import SAM2UNet

from utils.helper import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and test SAM2-UNet on Landslide Dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save predictions and metrics")
    parser.add_argument("--save_masks", type=lambda x: (str(x).lower() == 'true'), default=True, help="Save prediction PNGs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--blacklist_path", type=str, default=None, help="Path to blacklist txt file to skip corrupted data")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to dataset directory")
    return parser.parse_args()


def calculate_metrics(pred_prob: np.ndarray, gt_binary: np.ndarray, threshold: float = 0.5):
    """
    Computes standard segmentation metrics:
    IoU, Dice/F1, Precision, Recall, MAE, and mAP (Average Precision / PR-AUC).
    """
    pred_bin = (pred_prob >= threshold).astype(np.float32)
    gt = gt_binary.astype(np.float32)

    intersection = (pred_bin * gt).sum()
    union = (pred_bin + gt).clip(0, 1).sum()
    tp = intersection
    fp = (pred_bin * (1 - gt)).sum()
    fn = ((1 - pred_bin) * gt).sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (pred_bin.sum() + gt.sum() + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    mae = np.mean(np.abs(pred_prob - gt))

    gt_flat = gt.flatten().astype(np.int32)
    prob_flat = pred_prob.flatten().astype(np.float32)
    pos_count = int(gt_flat.sum())
    if pos_count == 0:
        ap = 1.0 if np.all(prob_flat < threshold) else 0.0
    elif pos_count == len(gt_flat):
        ap = 1.0 if np.all(prob_flat >= threshold) else 0.0
    else:
        try:
            ap = float(average_precision_score(gt_flat, prob_flat))
        except Exception:
            ap = 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "mae": float(mae),
        "map": float(ap),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Determine save directory
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        # Default save inside the run folder if checkpoint is in runs/
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        run_parent = os.path.dirname(ckpt_dir)
        save_dir = os.path.join(run_parent, f"{args.split}_results")

    probability_dir = os.path.join(save_dir, "probability_maps")
    binary_dir = os.path.join(save_dir, "binary_masks")
    if args.save_masks:
        os.makedirs(probability_dir, exist_ok=True)
        os.makedirs(binary_dir, exist_ok=True)
    print(f"Results and predictions will be saved to: {save_dir}")

    # Dataset & Dataloader
    ds_cfg = config["dataset"]
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    data_dir = args.data_dir if args.data_dir is not None else ds_cfg.get("data_dir", "datasets/landslide")
    blacklist_path = args.blacklist_path if args.blacklist_path is not None else ds_cfg.get("blacklist_path", "datasets/landslide/black_list.txt")
    if str(blacklist_path).lower() in ("none", "null", ""):
        blacklist_path = None
    target_size = ds_cfg.get("size", 352)

    if blacklist_path and os.path.exists(blacklist_path):
        from dataset import load_blacklist
        bl_set = load_blacklist(blacklist_path)
        print(f"Blacklist active: {len(bl_set)} corrupted/noisy samples will be skipped from '{blacklist_path}'")
    elif blacklist_path:
        print(f"Notice: Blacklist path specified but file not found: {blacklist_path}")

    dataset = LandslideDataset(
        data_dir=data_dir,
        split=args.split,
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
        mode="test",
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating {len(dataset)} valid samples from [{args.split}] split with modalities: {modalities}")

    # Model initialization
    topo_in_chans = sum(1 for m in modalities if m != "IMAGE")
    m_cfg = config.get("model", {})
    topo_backbone = m_cfg.get("topo_backbone", "convnext_tiny")
    use_kan = m_cfg.get("use_kan", True)

    print(f"Loading checkpoint weights from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if any("kan." in k for k in state_dict.keys()):
        use_kan = True

    model = SAM2UNet(
        checkpoint_path=None,
        mode = 'concatenation',
        attention_dsample = (2, 2),
        is_batchnorm = True,
        topo_in_chans=max(1, topo_in_chans),
        topo_backbone=topo_backbone,
        pretrained_topo=False,
        use_kan=use_kan,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating [{args.split}]"):
            images = batch["image"].to(device)
            labels = batch["label"]
            names = batch["name"]

            # Forward pass: take main output head
            preds, _, _ = model(images)

            # Resize logits only when model output and label sizes differ.
            target_size = labels.shape[-2:]
            if preds.shape[-2:] != target_size:
                preds_upsampled = F.interpolate(
                    preds,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                preds_upsampled = preds
            pred_probs = torch.sigmoid(preds_upsampled).cpu().numpy()

            for i in range(len(names)):
                name = names[i]
                prob = pred_probs[i, 0]  # Shape (H, W)
                gt = labels[i, 0].numpy()  # Shape (H, W), values {0, 1}

                m = calculate_metrics(prob, gt, threshold=args.threshold)
                all_metrics.append(m)

                if args.save_masks:
                    # Soft probability map: useful for calibration/inspection.
                    probability_img = (prob * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(probability_img).save(
                        os.path.join(probability_dir, f"{name}.png")
                    )

                    # Crisp binary mask: threshold is applied to the saved image.
                    binary_img = (prob >= args.threshold).astype(np.uint8) * 255
                    Image.fromarray(binary_img).save(
                        os.path.join(binary_dir, f"{name}.png")
                    )

    # Compute mean metrics across all samples
    avg_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0].keys()
    }

    # Print summary
    print("\n" + "=" * 50)
    print(f"EVALUATION METRICS SUMMARY [{args.split.upper()}] (N={len(all_metrics)})")
    print("=" * 50)
    for k, v in avg_metrics.items():
        print(f"  {k.upper():<12}: {v:.4f}")
    print("=" * 50)

    # Save metrics to text and JSON
    metrics_txt = os.path.join(save_dir, "metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Evaluation on Split: {args.split}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Modalities: {modalities}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        for k, v in avg_metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")

    metrics_json = os.path.join(save_dir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "checkpoint": args.checkpoint,
                "modalities": modalities,
                "threshold": args.threshold,
                "metrics": avg_metrics,
            },
            f,
            indent=2,
        )

    print(f"Metrics saved to:\n  - {metrics_txt}\n  - {metrics_json}")


if __name__ == "__main__":
    main()
