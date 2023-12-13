import pandas as pd
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm

result_imgs = glob('work_dirs/Pred_masks/*')
result = []

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

for img_path in tqdm(result_imgs):
    pred = Image.open(img_path)
    pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
    pred = np.array(pred) # 다시 수치로 변환
    # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
    for class_id in range(12):
        class_mask = (pred == class_id).astype(np.uint8)
        if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
            mask_rle = rle_encode(class_mask)
            result.append(mask_rle)
        else: # 마스크가 존재하지 않는 경우 -1
            result.append(-1)

submit = pd.read_csv('./data/SamsungDataset/sample_submission.csv')
submit['mask_rle'] = result
submit.to_csv('./baseline_submit.csv', index=False)
print('Done!')