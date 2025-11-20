###### describe anything seg_mask (mask_rle to mask) --> tokenize mask
# import cv2
# import torch
# import os
# import pandas as pd
# import random
# import numpy as np
# import torch
# from matplotlib.patches import Rectangle 
# from tqdm import tqdm
# import json
# from pycocotools import mask as mask_utils
# import gc
# import argparse

# #DOWNSAMPLE
# def downsample_mask_anyhit_area(mask: np.ndarray, target_size: tuple[int,int]) -> np.ndarray:
#     """target_size = (W_out, H_out). Returns 0/1 uint8 mask."""
#     W_out, H_out = target_size
#     m = cv2.resize(mask.astype(np.float32), (W_out, H_out), interpolation=cv2.INTER_AREA)
#     return (m > 0).astype(np.uint8)  # any contribution -> 1

# ## get df rows WITHOUT bboxes
# num_points = 4
# def get_bbox_or_point_df(df, num_points):
#     mask = df['bbox'].apply(
#         lambda x: (
#             isinstance(x, (list, tuple)) and len(x) != num_points ### changed to !=
#         ) or (
#             isinstance(x, np.ndarray) and x.ndim == 1 and x.size != num_points
#         )
#     )
#     return df[mask]

# N=13
# split = "train"
# save_dir = f"/data3/mgaur/mllm_inversion/seg_masks_182px_l14/{split}"
# os.makedirs(save_dir, exist_ok=True)

# data_dir = "/data3/mgaur"
# dataset_to_maskrle = {
#     "paco" : json.load(open(os.path.join(data_dir, "mllm_inversion/describe_anything/paco_81kbbox_mask_rle.json"), "r")),
#     "lvis" : json.load(open(os.path.join(data_dir, "mllm_inversion/describe_anything/lvis_373kbbox_mask_rle.json"), "r")),
#     "cocostuff" : json.load(open(os.path.join(data_dir, "mllm_inversion/describe_anything/cocostuff_32kbbox_mask_rle.json"), "r")),
#     "mapillary" : json.load(open(os.path.join(data_dir, "mllm_inversion/describe_anything/mapillary_100kbbox_mask_rle.json"), "r")),
# }

# df_path = f"/data3/mgaur/mllm_inversion/datasets/combined_df_2.4m_{split}_preproc_fixed_bbox_format_no_paco_no_cocostuff.parquet"
# FULL_DF = pd.read_parquet(df_path)
# for _ in FULL_DF.bbox.tolist():
#     if isinstance(_, str):
#         raise Exception("dataframe has some bboxes stored in str format")

# df = get_bbox_or_point_df(FULL_DF, num_points)
# num_samples = len(df)

# #### plug and play code to partition df

# parser = argparse.ArgumentParser()
# parser.add_argument("--partition_idx", type=int, required=True, help="Which partition to run on (0..4)")
# args = parser.parse_args()

# num_partitions = 15
# parts = np.array_split(df, num_partitions)  # 5 roughly equal partitions
# if not (0 <= args.partition_idx < num_partitions):
#     raise ValueError(f"partition_idx must be in [0, 4], got {args.partition_idx}")
# df = parts[args.partition_idx].copy()
# num_samples = len(df)  # keep tqdm total correct
# ####### 

# for _, row in enumerate(tqdm(df.itertuples(), total = num_samples)):
#     if _ % 1000==0:
#         gc.collect()
#     idx = row.Index    #row based idx, not integer position based
#     save_path = os.path.join(save_dir, f"{idx}.pt")
#     if os.path.isfile(save_path):
#         continue

#     row = df.loc[idx]
#     dataset = row['dataset']
#     dataset_sample_idx = row['dataset_sample_idx']
#     mask_rle_list = dataset_to_maskrle[dataset.split(" ")[0]]
#     mask_rle = mask_rle_list[dataset_sample_idx]
#     gt_binary_mask = mask_utils.decode(mask_rle) # boolean seg_mask (H,W)
#     tokenized_mask = downsample_mask_anyhit_area(gt_binary_mask, (N,N))

#     torch.save(torch.from_numpy(tokenized_mask), save_path)

##############################################################################################################################################################################################################################################################################
###### sam masks curated from bbox --> tokenized_mask
######## saves one torch tensor (num_df_samples, N*N) uint8

import cv2
import torch
import os
import pandas as pd
import random
import numpy as np
import torch
from matplotlib.patches import Rectangle 
from tqdm import tqdm
import argparse

def downsample_mask_anyhit_area(mask: np.ndarray, target_size: tuple[int,int]) -> np.ndarray:
    """target_size = (W_out, H_out). Returns 0/1 uint8 mask."""
    W_out, H_out = target_size
    m = cv2.resize(mask.astype(np.float32), (W_out, H_out), interpolation=cv2.INTER_AREA)
    return (m > 0).astype(np.uint8)  # any contribution -> 1


num_points = 4
def get_bbox_or_point_df(df, num_points):
    mask = df['bbox'].apply(
        lambda x: (
            isinstance(x, (list, tuple)) and len(x) == num_points
        ) or (
            isinstance(x, np.ndarray) and x.ndim == 1 and x.size == num_points
        )
    )
    return df[mask]

N=13 # img_res // patch_size
split = "val"
save_dir = f"/data3/mgaur/mllm_inversion/seg_masks_182px_l14/{split}"
os.makedirs(save_dir, exist_ok=True)


df_path = f"/data3/mgaur/mllm_inversion/datasets/combined_df_2.4m_{split}_preproc_fixed_bbox_format_no_paco_no_cocostuff.parquet"
FULL_DF = pd.read_parquet(df_path)
for _ in FULL_DF.bbox.tolist():
    if isinstance(_, str):
        raise Exception("dataframe has some bboxes stored in str format")

df = get_bbox_or_point_df(FULL_DF, num_points)
num_samples = len(df)
print(num_samples)

missing = [] #missing seg_masks
# all_masks = np.zeros((num_samples, N * N), dtype=np.uint8)
seg_mask_dir = f"/data3/mgaur/mllm_inversion/seg_masks/{split}"


# #### plug and play code to partition df
# parser = argparse.ArgumentParser()
# parser.add_argument("--partition_idx", type=int, required=True, help="Which partition to run on (0..4)")
# args = parser.parse_args()

# parts = np.array_split(df, 5)  # 5 roughly equal partitions
# if not (0 <= args.partition_idx < 5):
#     raise ValueError(f"partition_idx must be in [0, 4], got {args.partition_idx}")
# df = parts[args.partition_idx].copy()
# num_samples = len(df)  # keep tqdm total correct
#############
# for row in tqdm(df.iloc[::-1].itertuples(), total = num_samples):
for row in tqdm(df.itertuples(), total = num_samples):
    idx = row.Index
    seg_path = os.path.join(seg_mask_dir, f"{idx}.pt")
    # if not os.path.isfile(seg_path):
    #     missing.append(seg_path)
    save_path = os.path.join(save_dir, f"{idx}.pt")
    if os.path.isfile(save_path):
        continue
    gt_mask = torch.load(seg_path).numpy().astype(np.uint8)
    tokenized_mask = downsample_mask_anyhit_area(gt_mask, (N,N))
    
    # all_masks[idx] = tokenized_mask.reshape(-1)
    # torch.save(torch.from_numpy(all_masks), os.path.join(save_dir, f"{split}.pt"))
    
    torch.save(torch.from_numpy(tokenized_mask),save_path)