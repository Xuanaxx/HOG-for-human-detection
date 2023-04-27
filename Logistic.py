from skimage import feature
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import joblib
import cv2
import numpy as np
from PIL import Image
import random


class HOG:
    def __init__(self, orientations=9, pixelsPerCell=(8, 8),
                 cellsPerBlock=(2, 2), transform=False):
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform

    def describe(self, image):
        hist = feature.hog(image, orientations=self.orienations,
                           pixels_per_cell=self.pixelsPerCell,
                           cells_per_block=self.cellsPerBlock,
                           transform_sqrt=self.transform)

        return hist


def load_images(dirname, amout=9999):
    img_list = []
    file = open(dirname)
    img_name = file.readline()
    while img_name != '':  # 文件尾
        img_name = dirname.rsplit(r'/', 1)[0] + r'/' + img_name.split('/', 1)[1].strip('\n')
        img_list.append(cv2.imread(img_name))
        img_name = file.readline()
        amout -= 1
        if amout <= 0:  # 控制读取图片的数量
            break
    return img_list


# 从每一张没有人的原始图片中随机裁出10张64*128的图片作为负样本
def sample_neg(full_neg_lst, neg_list, size):
    random.seed(1)
    width, height = size[1], size[0]
    for i in range(len(full_neg_lst)):
        for j in range(10):
            y = int(random.random() * (len(full_neg_lst[i]) - height))
            x = int(random.random() * (len(full_neg_lst[i][0]) - width))
            neg_list.append(full_neg_lst[i][y:y + height, x:x + width])
    return neg_list


# wsize: 处理图片大小，通常64*128; 输入图片尺寸>= wsize
def computeHOGs(img_lst, gradient_lst, wsize=(128, 64)):
    hog = HOG(transform=True)  # 实例化一个HOG类
    # hog.winSize = wsize
    for i in range(len(img_lst)):
        if img_lst[i].shape[1] >= wsize[1] and img_lst[i].shape[0] >= wsize[0]:
            roi = img_lst[i][(img_lst[i].shape[0] - wsize[0]) // 2: (img_lst[i].shape[0] - wsize[0]) // 2 + wsize[0], \
                  (img_lst[i].shape[1] - wsize[1]) // 2: (img_lst[i].shape[1] - wsize[1]) // 2 + wsize[1]]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gradient_lst.append(hog.describe(gray))  # hog.compute(gray)为3780*1维向量,gradient_lst为3780*2416维数组
    # return gradient_lst


# 主程序
# 第一步：计算HOG特征
neg_list = []
pos_list = []
gradient_lst = []
labels = []
pos_list = load_images(r'D:/Desktop/python_work/MachineLearning/INRIAPerson/normalized_images/train/pos.lst')
full_neg_lst = load_images(
    r'D:/Desktop/python_work/MachineLearning/INRIAPerson/normalized_images/train/neg.lst')
sample_neg(full_neg_lst, neg_list, [128, 64])
print(len(neg_list))
computeHOGs(pos_list, gradient_lst)
[labels.append(+1) for _ in range(len(pos_list))]  # pos_list=2416
computeHOGs(neg_list, gradient_lst)
[labels.append(0) for _ in range(len(neg_list))]  # neg_list=12180（每张neg图片crop10，有1218张图片）,label为14596

print("### Training model...")
model = LogisticRegression(max_iter=5000)  # 利用sklearn中的linear_model进行训练
model.fit(np.array(gradient_lst), np.array(labels))
joblib.dump(model, "modelLog")
