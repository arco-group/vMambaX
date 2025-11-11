# Modified by emulero 10/2025
# Copyright (c) 2025 Jie Mei

import os
import numpy as np
import torch
import cv2
import torch.nn as nn
from src.models.builder import EncoderDecoder as segmodel
import argparse
from medpy.metric.binary import hd95
from tqdm import tqdm
from torch.serialization import add_safe_globals
from pathlib import Path

add_safe_globals([argparse.Namespace])

def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(path, map_location=device)

    if path.suffix == ".ckpt":
        state_dict = checkpoint.get("state_dict", checkpoint)
        filtered_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
        if filtered_state_dict:
            state_dict = filtered_state_dict
    else:
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model

def processImage(pet_path, ct_path, mask_path, model, outPath, image_id, total_tp, total_fp, total_fn, total_tn, hd95_list, device):
    model.eval()
    pet_img = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)  #pet
    ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)  #ct
    # print(image_id)

    ct_img = np.expand_dims(ct_img, axis=2)
    pet_img = np.expand_dims(pet_img, axis=2)
    ct_img = np.array(ct_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    pet_img = np.array(pet_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    pet_img = pet_img[np.newaxis, :, :, :]
    ct_img = ct_img[np.newaxis, :, :, :]
    pet_img = torch.tensor(pet_img, dtype=torch.float32, device=device).repeat(1, 3, 1, 1)
    ct_img = torch.tensor(ct_img, dtype=torch.float32, device=device).repeat(1, 3, 1, 1)
    model.to(device)
    with torch.no_grad():
        pred =  torch.sigmoid(model.forward(ct_img,pet_img))
    pred = pred.cpu().numpy()
    pred = np.squeeze(pred, axis=0).transpose(1, 2, 0)*255.0
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite(os.path.join(os.path.dirname(pet_path), image_id + '_proposal_mask_cipa' + '.png'), pred)

    gt_bin = (mask_img > 255*0.5).astype(np.uint8)
    pred_bin = (pred[:,:,0] > 255*0.5).astype(np.uint8)

    tp = np.sum((gt_bin == 1) & (pred_bin == 1))
    fp = np.sum((gt_bin == 0) & (pred_bin == 1))
    fn = np.sum((gt_bin == 1) & (pred_bin == 0))
    tn = np.sum((gt_bin == 0) & (pred_bin == 0))

    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        hd = 0.0
    elif gt_bin.sum() == 0 or pred_bin.sum() == 0:
        pred_bin[256, 256]=1
        hd = hd95(pred_bin, gt_bin)
    else:
        try:
            hd = hd95(pred_bin, gt_bin)
        except:
            hd = np.sqrt(gt_bin.shape[0]**2 + gt_bin.shape[1]**2)
    hd95_list.append(hd)

    return total_tp, total_fp, total_fn, total_tn, hd95_list


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)
    model = load_checkpoint_into_model(model, args.checkpoint, device)
    model.to(device)
    with open(os.path.join(args.split_train_val_test, 'test.txt'), 'r') as f:
        test_list = [x[:-1] for x in f]
    outPath="./results/"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    hd95_list = []
    for image_id in tqdm(test_list):
        pet_path = os.path.join(args.img_dir, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, "_PET.png")
        ct_path = os.path.join(args.img_dir, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, "_CT.png")
        mask_path = os.path.join(args.img_dir, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, "_mask.png")
        total_tp, total_fp, total_fn, total_tn, hd95_list = processImage(
            pet_path,
            ct_path,
            mask_path,
            model,
            outPath,
            image_id,
            total_tp,
            total_fp,
            total_fn,
            total_tn,
            hd95_list,
            device,
        )

    total_pixels = total_tp + total_fp + total_fn + total_tn
    # IoU
    denominator_iou = total_tp + total_fp + total_fn
    iou = 1.0 if denominator_iou == 0 else total_tp / denominator_iou
    # Dice/F1
    denominator_dice = 2 * total_tp + total_fp + total_fn
    dice = 1.0 if denominator_dice == 0 else (2 * total_tp) / denominator_dice
    # Accuracy
    acc = np.mean([total_tp/(total_tp+total_fn), total_tn/(total_tn+total_fp)])
    # HD95
    mean_hd95 = np.mean(hd95_list)
    print(f"IoU: {iou:.6f}")
    print(f"Dice: {dice:.6f}")
    print(f"Acc: {acc:.6f}")
    print(f"HD95: {mean_hd95:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--device", default="cuda", help="training device")

    parser.add_argument("--img_dir", type=str, help='Path to the dataset')
    parser.add_argument("--split_train_val_test", type=str, 
                        help='Path to the train.txt and test.txt files with the split sets')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=os.path.join('save_model', 'CIPA.pth'),
                        help="path to trained model checkpoint (.pth or .ckpt)")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
