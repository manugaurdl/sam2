from transformers import Sam2Processor, Sam2Model
import torch
from PIL import Image
from accelerate import Accelerator
from tqdm import tqdm

device = Accelerator().device
model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")


import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

if os.path.isdir("/storage/users/manugaur"):
    data_dir = "/storage/users/manugaur"
elif os.path.isdir("/workspace/manugaur"):
    data_dir = "/workspace/manugaur"
elif os.path.isdir("/data3/mgaur/mllm_inversion"):
    data_dir = "/data3/mgaur/mllm_inversion"
else:
    raise Exception("input data dir bruh")

def get_coco_path(cocoid_img_name: str) -> str:
    if ".jpg" not in cocoid_img_name:
        cocoid_img_name += ".jpg"
    image_path = os.path.join(
        data_dir, "coco", cocoid_img_name.split("_")[1], cocoid_img_name
    )
    if os.path.isfile(image_path):
        return image_path
    raise Exception("COCO image not found")

def get_vg_path(vg_id: str) -> str:
    img_name = vg_id + ".jpg"
    pathA = os.path.join(data_dir, "visual_genome_non_coco/VG_100K_2", img_name)
    pathB = os.path.join(data_dir, "visual_genome_non_coco/VG_100K", img_name)
    if os.path.isfile(pathA):
        return pathA
    if os.path.isfile(pathB):
        return pathB
    raise Exception("VG image not found")

def get_mapillary_path(uid: str) -> str:
    img_path = os.path.join(data_dir, "mapillary/images", uid + ".jpg")
    if os.path.isfile(img_path):
        return img_path
    raise Exception(f"{img_path} not found")

def get_bbox_df(df):
    mask = df['bbox'].apply(
        lambda x: (
            isinstance(x, (list, tuple)) and len(x) == 4
        ) or (
            isinstance(x, np.ndarray) and x.ndim == 1 and x.size == 4
        )
    )
    
    return df[mask]          # preserves the original native index

class ImageBoxDataset(Dataset):
    """
    Returns:
        image_rgb: np.ndarray (H, W, 3), dtype=uint8
        boxes:     np.ndarray (1, 4) in [x1, y1, x2, y2]
    """
    def __init__(
        self, 
        split = "val",
        num_shards = 1,
        shard_idx = 0,
    ):
        parquet_path = os.path.join(
            data_dir,
            f"datasets/combined_df_2.4m_{split}_preproc_fixed_bbox_format_no_paco_no_cocostuff.parquet",
        )
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")
        FULL_DF = pd.read_parquet(parquet_path)
        for _ in FULL_DF.bbox.tolist():
            if isinstance(_, str):
                raise Exception("dataframe has some bboxes stored in str format")
        self.df = get_bbox_df(FULL_DF)
        
        #sharding dataframe indices
        n = len(self.df)
        if not (0 <= shard_idx < num_shards):
            raise ValueError(f"shard_idx must be in [0, {num_shards-1}], got {shard_idx}")

        base = n // num_shards            # minimum rows per shard
        rem  = n % num_shards             # extra rows to give to shard 0

        # first shard gets the remainder
        self._shard_len = base + (rem if shard_idx == 0 else 0)
        # start offset: shard 0 starts at 0; rest start at (shard_idx * base + rem)
        self._shard_start = 0 if shard_idx == 0 else (rem + shard_idx * base)

        # store for convenience / debugging
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.total_len = n

    def __len__(self):
        return self._shard_len

    def __getitem__(self, idx):
        
        idx = self._shard_start + idx

        # uid, ref, bbox, cocoid, dataset, dataset_sample_idx, mask_rle_box, cap_len, og_df_index = tuple(self.df.iloc[idx])
        row = self.df.iloc[idx]
        uid      = row["uid"]
        bbox     = row["bbox"]
        cocoid   = row["cocoid"]
        dataset  = row["dataset"]

        # Pick image path
        if cocoid is not None:
            img_path = get_coco_path(cocoid)
        elif dataset == "visual genome":
            img_path = get_vg_path(uid)
        elif dataset == "mapillary DA":
            img_path = get_mapillary_path(uid)
        else:
            raise Exception("__getitem__ error: image from unknown dataset")

        # Boxes -> [x1, y1, x2, y2]
        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
            x, y, w, h = bbox
            x_max = x + w
            y_max = y + h
            boxes = np.array([x, y, x_max, y_max], dtype=float).reshape(1, 4)
        else:
            # Assume already [x1,y1,x2,y2] or (N,4)
            boxes = np.array(bbox, dtype=float)
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, 4)

        # Read image (BGR -> RGB)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image_rgb, boxes, idx

def collate_images_boxes(batch):
    # batch is list of tuples: [(image_rgb, boxes), ...]
    images = [item[0] for item in batch]
    boxes  = [item[1] for item in batch]
    idx  = [item[2] for item in batch]
    return images, boxes, idx 

def main(args):
    save_dir = args.save_dir
    split = args.split
    batch_size = args.batch_size
    num_shards = args.num_shards
    shard_idx = args.shard_idx
    #### df path hardcoded ; data_dir hardcoded!!!
    os.makedirs(os.path.join(save_dir, split), exist_ok=True)
    
    
    ds = ImageBoxDataset(split=split, num_shards=num_shards, shard_idx=shard_idx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_images_boxes)


    for images, boxes, df_idx in tqdm(dl, total=len(dl)):
        inputs = processor(images=images, input_boxes=boxes, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])

        for idx in range(len(masks)):
            mask = masks[idx]
            binary_mask = mask[0].sum(0)
            final_mask = torch.where(binary_mask!= 0, 1, binary_mask)
            torch.save(final_mask.to(torch.int8), os.path.join(save_dir, split, f"{df_idx[idx]}.pt"))

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, help="train and val folders will be created here", required=True) 
    parser.add_argument("--split", type=str, choices=["train", "val"], required=True)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_shards", type=int, help='how many shards to create from dataset', default=1)
    parser.add_argument("--shard_idx", type=int, help='which shard to operate on', default=0)
    args = parser.parse_args()
    
    main(args)