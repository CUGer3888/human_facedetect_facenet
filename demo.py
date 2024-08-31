import torch
from torch import nn
import cv2
import numpy as np
from PIL import Image
from modle import FaceNetmModel
from mtcnn.mtcnn import MTCNN
from glob import glob
import math
from torch.nn import functional as F
import torchvision.transforms as transforms
detector = MTCNN()
model = FaceNetmModel()


def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def face_cor(img):
    try:
        face = detector.detect_faces(img)
        bbox, land_mask = face[0]['box'], face[0]['keypoints']
    except:
        img = cv_to_pil(img)
        face = detector.detect_faces(img)
        bbox, land_mask = face[0]['box'], face[0]['keypoints']

    left_eye = [land_mask['left_eye'][0], land_mask['left_eye'][1]]
    right_eye = [land_mask['right_eye'][0], land_mask['right_eye'][1]]

    k = (right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])
    angle = math.degrees(math.atan(k))
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.rotate(angle, expand=True)
    print(bbox)
    face_img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    return pil_to_cv(face_img)


def face_net(img1, img2):
    pred_arr1, mpead_arr2 = model(img1), model(img2)
    return F.pairwise_distance(pred_arr1, mpead_arr2)


def pridict(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceNetmModel(256, 108)
    model.load_state_dict(torch.load('facenet.pth'))
    model = model.to(device)

    model.eval()  # 预测模式

    # 获取测试图片，并行相应的处理
    img = cv2.imread(path)
    img = face_cor(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    transform = transforms.Compose([transforms.Resize(256),  # 重置图像分辨率
                                    transforms.CenterCrop(128),  # 中心裁剪
                                    transforms.ToTensor(), ])
    img = Image.fromarray(img)
    img = img.convert("RGB")  # 如果是标准的RGB格式，则可以不加
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        py = model(img)
    predicted, _ = torch.max(py, 1)  # 获取分类结果
    classIndex_ = predicted[0]

    print('预测结果', classIndex_)
# 预测结果 tensor(8.5014, device='cuda:0')
# 预测结果 tensor(8.1327, device='cuda:0')
# 效果还不错
if __name__ == '__main__':
    pridict('test1.jpg')
