import cv2
import json
import numpy as np
from math import sqrt
from glob import glob
from tqdm import tqdm

def get_fish_xn_yn(source_x, source_y, radius, distortion):

    if 1 - distortion*(radius**2) == 0:
        return source_x, source_y

    return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))

def fisheye_effect(img, distortion_coefficient):

    w, h = img.shape[0], img.shape[1]
    dstimg = np.full_like(img, 12)  # Initialize with 12
    w, h = float(w), float(h)
    maxx, maxy = 0, 0

    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)
            rd = sqrt(xnd**2 + ynd**2)
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)
            xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]
                if maxx < x: maxx = x
                if maxy < y: maxy = y
    
    maxx, maxy = maxx - 80, maxy - 100
    cropedimg = dstimg[int(w) - maxx:maxx + 1, int(h) - maxy:maxy + 1]
    
    return cropedimg.astype(np.uint8)

def resize_image(img, target_size):
    return cv2.resize(img, target_size)

def convert_to_grayscale(input_path):
    # 이미지 로드
    img = cv2.imread(input_path)

    # RGB 이미지를 그레이스케일로 변환
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 결과 이미지 저장
    cv2.imwrite(input_path, gray_img)

def mask_roi_image(source_img: np.ndarray, target_img_path, labelme_json_path, is_mask) -> np.ndarray:
    """
    이미지에 대해 주어진 다각형 좌표에 대한 마스크를 생성하여 이미지를 마스킹한 결과를 반환하는 함수입니다.

    Args:
        image (np.ndarray): 마스킹할 대상 이미지.
        iou_list (List[Tuple[int, int]]): 다각형의 좌표 리스트.

    Returns:
        np.ndarray: 다각형에 대한 마스크가 적용된 이미지.
    """
    # 이미지 불러오기
    source_img = source_img # 기존 이미지
    target_img = cv2.imread(target_img_path) # 마스크 Background

    # labelme JSON 파일 읽기
    with open(labelme_json_path, 'r') as json_file:
        labelme_data = json.load(json_file)

    # ROI 정보 추출
    shapes = labelme_data['shapes']
    polygons = [np.array(shape['points'], dtype=np.int32) for shape in shapes]

    mask = np.zeros_like(target_img)

    ignore_mask_color = (255,) * target_img.shape[-1] if len(target_img.shape) > 2 else 255
    shape = polygons
    cv2.fillPoly(mask, shape, ignore_mask_color)
    inverted_mask = cv2.bitwise_not(mask)

    # ROI를 다른 이미지에 덮어 씌우기
    result_img = cv2.bitwise_and(target_img, inverted_mask)

    # 이미지 크기 맞추기
    result_img = resize_image(result_img, source_img.shape[1::-1])
    mask = resize_image(mask, source_img.shape[1::-1])
    
    source_img = cv2.bitwise_and(source_img, mask)

    if is_mask:
        # ROI 바깥 영역을 12로 채우기
        outside_roi = np.full_like(source_img, 12)
        inverted_mask = resize_image(inverted_mask, source_img.shape[1::-1])
        outside_roi = cv2.bitwise_and(outside_roi, inverted_mask)
        masked_image = cv2.add(outside_roi, source_img)
        masked_image[masked_image > 12] = 12
    else:
        masked_image = cv2.add(result_img, source_img)

    return masked_image

if __name__ == "__main__":

    img_paths = glob('/home/steven6774/hdd/InternImage/segmentation/data/SamsungDataset/images/*/*') + glob('/home/steven6774/hdd/InternImage/segmentation/data/SamsungDataset/annotations/*/*')
    backgrounds = glob('/home/steven6774/hdd/InternImage/segmentation/data/SamsungDataset/background/*.json')

    for img_path in tqdm(img_paths):

        imgobj = cv2.imread(img_path)

        fisheye_img = fisheye_effect(imgobj, 1.0)

        labelme_json_path = backgrounds[int(img_path.split('/')[-1].split('.')[0].split('_')[-1]) % 2]

        if len(np.unique(fisheye_img)) < 15:
            fisheye_img[fisheye_img == 255] = 12
            # print(np.unique(fisheye_img))

        fisheye_img = mask_roi_image(fisheye_img, labelme_json_path.replace('json', 'png'), labelme_json_path, len(np.unique(fisheye_img)) < 15)

        cv2.imwrite(img_path, fisheye_img)
    
    for mask in tqdm(glob('/home/steven6774/hdd/InternImage/segmentation/data/SamsungDataset/annotations/*/*')):
        convert_to_grayscale(mask)