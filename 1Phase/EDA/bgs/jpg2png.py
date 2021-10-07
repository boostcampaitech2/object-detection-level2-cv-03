from PIL import Image
from glob import glob
from tqdm import tqdm

all_path = glob('./bgPatch/*')
print(len(all_path))

for path in tqdm(all_path):
    fname = path.split('/')[-1].split('.')[-2]
    img = Image.open(path)
    img.save(f'./bgPatchPNG/{fname}.png')