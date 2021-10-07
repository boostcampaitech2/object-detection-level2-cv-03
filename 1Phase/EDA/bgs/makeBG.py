from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# from random import randint, gauss
import random

tomake = 5000
dstroot = './BG'

all_patches = glob('./bgPatchPNG/*')
n_tot = len(all_patches)
print(n_tot)

pt_h, pt_v = 0, 0

dense = 32
anchors = []
for i in range(0, 1024+1, dense):
    for j in range(0, 1024+1, dense):
        anchors.append((i,j))

paper = np.zeros((1024,1024,3))
paper = Image.fromarray(paper.astype(np.uint8))
for idx in tqdm(range(tomake)):
    for ax, ay in anchors:
        patch = Image.open(all_patches[random.randint(0,n_tot-1)])
        p_w, p_h = patch.size
        patch_x, patch_y = int(ax-p_w/2), int(ay-p_h/2)
        
        if not 0<= patch_x <= 1024:
            patch_x = 0
        if not 0<= patch_y <= 1024:
            patch_y = 0
        random.shuffle(anchors)

        
        paper.paste(patch, (patch_x, patch_y))

    paper.save(f'./BG/{idx}.jpg')