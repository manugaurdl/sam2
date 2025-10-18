import gc
from tqdm import tqdm
import os
import torch

stacked = []
split = 'val'
tokenized_mask_dir = f"/data3/mgaur/mllm_inversion/seg_masks_336px_l14/{split}"
files = os.listdir(tokenized_mask_dir)

for idx in tqdm(range(len(files)), total=len(files)):
    if idx%1000==0:
        torch.cuda.empty_cache()
        gc.collect()
    mask = torch.load(os.path.join(tokenized_mask_dir, f"{idx}.pt")).view(-1)
    stacked.append(mask)

torch.save(torch.stack(stacked), f"/data3/mgaur/mllm_inversion/seg_masks_336px_l14/{split}.pt")


"""
df is monotonically increasing.
but when i create subsets (bbox vs 2 points), then df.loc and iloc don't match
I create seg masks separately for both splits (4 and 2 points)
I store file names using df.loc (which is the same as  FULL_OG_DF.iloc)
so in dataset.__getitem__, i can use df.iloc :)
"""