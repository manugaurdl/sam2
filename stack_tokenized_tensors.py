import gc
from tqdm import tqdm
import os
import math
import torch
import pandas as pd

# -------- config --------
split = 'train'
num_partitions = 20  # <-- set N here
base_dir = "/data3/mgaur/mllm_inversion/seg_masks_182px_l14"
tokenized_mask_dir = f"{base_dir}/{split}"
final_out_path = f"{base_dir}/{split}.pt"
# ------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--partition_ids", 
    type=int, 
    nargs='+',  # This is the key
    required=True, 
    help="Which partition(s) to run on (e.g., 0 1 2)"
)
args = parser.parse_args()

df_path = f"/data3/mgaur/mllm_inversion/datasets/combined_df_2.4m_{split}_preproc_fixed_bbox_format_no_paco_no_cocostuff.parquet"
df = pd.read_parquet(df_path)
print(f"df len {len(df)}")

files = [f for f in os.listdir(tokenized_mask_dir) if f.endswith('.pt')]
print(f"{len(files)} tokenized masks exist")
assert len(files) == len(df), "files count must match df length"

total = len(files)
assert total > 0, "No files found to stack."

# Determine feature dimension & dtype from the first mask
first_vec = torch.load(os.path.join(tokenized_mask_dir, "0.pt")).view(-1)
feat_dim = first_vec.numel()
feat_dtype = first_vec.dtype
print(f"Detected feature dim: {feat_dim}, dtype: {feat_dtype}")

# ---------------- Part 1: create N partition stacks ----------------
part_size = math.ceil(total / num_partitions)
print(f"Creating {num_partitions} partitions of size ~{part_size}")

# for p in range(num_partitions):
for p in args.partition_ids:
    part_path = f"{base_dir}/{split}_partition{p}.pt"
    if os.path.isfile(part_path):
        continue
    print(f"|stacking PARTITION ID {p}...")
    start = p * part_size
    end = min((p + 1) * part_size, total)
    if start >= end:
        break  # in case num_partitions > total

    stacked = []
    rng = range(start, end)
    for idx in tqdm(rng, total=(end - start), desc=f"Partition {p}/{num_partitions-1}"):
        if idx % 1000 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        mask = torch.load(os.path.join(tokenized_mask_dir, f"{idx}.pt")).view(-1)
        if mask.numel() != feat_dim:
            raise ValueError(f"Mask {idx} has dim {mask.numel()} != expected {feat_dim}")
        # keep dtype consistent
        if mask.dtype != feat_dtype:
            mask = mask.to(feat_dtype)
        stacked.append(mask)

    part_tensor = torch.stack(stacked) if len(stacked) > 0 else torch.empty((0, feat_dim), dtype=feat_dtype)
    torch.save(part_tensor, part_path)
    print(f"Saved partition {p}: shape {tuple(part_tensor.shape)} -> {part_path}")
    del stacked, part_tensor
    gc.collect()

# ---------------- Part 2: stitch partitions into one big stack ----------------
# print("Stitching all partitions into the final stacked tensor...")
# final = torch.empty((total, feat_dim), dtype=feat_dtype)

# write_ptr = 0
# for p in range(num_partitions):
#     part_path = f"{base_dir}/{split}_partition{p}.pt"
#     if not os.path.isfile(part_path):
#         # Skip non-existent partitions (e.g., when total < num_partitions * part_size for the tail)
#         continue
#     part_tensor = torch.load(part_path)
#     sz = part_tensor.shape[0]
#     if sz == 0:
#         continue
#     final[write_ptr:write_ptr + sz] = part_tensor
#     write_ptr += sz
#     print(f"Loaded partition {p}: shape {tuple(part_tensor.shape)} -> wrote rows [{write_ptr - sz}, {write_ptr})")
#     del part_tensor
#     gc.collect()

# if write_ptr != total:
#     raise RuntimeError(f"Final write_ptr {write_ptr} != total {total}. "
#                        "Partitioning or file set may be inconsistent.")

# torch.save(final, final_out_path)
# print(f"|SAVED final tensor {tuple(final.shape)} -> {final_out_path}")




# import gc
# from tqdm import tqdm
# import os
# import torch
# import pandas as pd


# split = 'train'
# df_path = f"/data3/mgaur/mllm_inversion/datasets/combined_df_2.4m_{split}_preproc_fixed_bbox_format_no_paco_no_cocostuff.parquet"
# df = pd.read_parquet(df_path)
# print(f"df len {len(df)}")

# tokenized_mask_dir = f"/data3/mgaur/mllm_inversion/seg_masks_182px_l14/{split}"
# files = [f for f in os.listdir(tokenized_mask_dir) if f.endswith('.pt')]
# print(f"{len(files)} tokenized masks exist")
# assert len(files)==len(df)

# stacked = []
# for idx in tqdm(range(len(files)), total=len(files)):
#     if idx%1000==0:
#         torch.cuda.empty_cache()
#         gc.collect()
#     mask = torch.load(os.path.join(tokenized_mask_dir, f"{idx}.pt")).view(-1)
#     stacked.append(mask)
# out = torch.stack(stacked)
# print(f"|SAVING {out.shape} tensor")
# torch.save(out, f"/data3/mgaur/mllm_inversion/seg_masks_182px_l14/{split}.pt")

# """
# df is monotonically increasing.
# but when i create subsets (bbox vs 2 points), then df.loc and iloc don't match
# Hence, I had to create seg masks separately for both splits (bbox and pointing)
# I store file names using df.loc (which is the same as  FULL_OG_DF.iloc)
# so in dataset.__getitem__, i can use df.iloc :)
# """