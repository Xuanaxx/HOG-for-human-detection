import cv2
import numpy as np
import pandas as pd
import numpy as np
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from skimage import color
import joblib


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


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield x, y, image[y: y + window_size[1], x:x + window_size[0]]


def detector(pos_list, neg_list):
    # 最小窗口大小
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    # 图像金字塔缩放倍数
    downscale = 1.25
    # 导入训练好的Log模型
    clf = joblib.load("modelLog")
    # 对图片进行缩放
    TP = np.zeros([1, 15])
    FN = np.zeros([1, 15])
    FP = np.zeros([1, 15])
    TN = np.zeros([1, 15])
    counter = 0
    for hitThreshold in [k / 2 for k in range(-11, 4, 1)]:
        num_pos = 0
        for i in range(len(neg_list)):
            num_pos += 1
            print(str(num_pos) + '/' + str(len(neg_list)) + '‐' + str(hitThreshold))
            # 对图片大小进行控制
            neg_list[i] = imutils.resize(neg_list[i], width=min(400, neg_list[i].shape[1]))
            for im_scaled in pyramid_gaussian(neg_list[i], downscale=downscale):
                # 如果图片比我们规定的图片小，那么就结束，不进行下面的hog检测
                if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                    break
                # 对滑动窗口进行hog特征检测
                for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                    if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                        continue
                    # 将得到的图像进行灰度化
                    im_window = color.rgb2gray(im_window)
                    # hog特征检测
                    fd = hog(im_window, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                             transform_sqrt=True)
                    # 将fd特征转换成向量，并调用训练好的模型进行预测
                    fd = fd.reshape(1, -1)
                    # 如果检测到图片中有人
                    flag = False
                    if clf.decision_function(fd) > hitThreshold:  # 预测样本属于positive正样本的的概率大于阈值，判定为有人
                        FP[0, counter] += 1
                        flag = True  # True表示在该层高斯金字塔识别出人
                        break
                    else:
                        continue
                if flag != True:  # 在该层高斯金字塔没识别出人
                    continue  # 下一层高斯金字塔
                else:
                    break
            if flag != True:
                TN[0, counter] += 1
        num_pos = 0
        for i in range(len(pos_list)):
            num_pos += 1
            print(str(num_pos) + '/' + str(len(pos_list)) + '‐' + str(hitThreshold))
            # 对图片大小进行控制
            pos_list[i] = imutils.resize(pos_list[i], width=min(400, pos_list[i].shape[1]))
            for im_scaled in pyramid_gaussian(pos_list[i], downscale=downscale):
                # 如果图片比我们规定的图片小，那么就结束，不进行下面的hog检测
                if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                    break
                # 对滑动窗口进行hog特征检测
                for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                    if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                        continue
                    # 将得到的图像进行灰度化
                    im_window = color.rgb2gray(im_window)
                    # hog特征检测
                    fd = hog(im_window, orientations=9, pixels_per_cell=[8, 8], cells_per_block=[2, 2],
                             transform_sqrt=True)
                    # 将fd特征转换成向量，并调用训练好的模型进行预测
                    fd = fd.reshape(1, -1)
                    # 如果检测到图片中有人
                    flag = False
                    if clf.decision_function(fd) > hitThreshold:  # 预测样本属于positive正样本的的概率大于阈值，判定为有人
                        TP[0, counter] += 1
                        flag = True  # True表示在该层高斯金字塔识别出人
                        break
                    else:
                        continue
                if flag != True:  # 在该层高斯金字塔没识别出人
                    continue  # 下一层高斯金字塔
                else:
                    break
            if flag != True:
                FN[0, counter] += 1
        counter += 1
    return [TP, FN, FP, TN]


neg_list = []
pos_list = []
pos_list = load_images(r'D:/Desktop/python_work/MachineLearning/INRIAPerson/original_images/Test/pos.lst')
neg_list = load_images(
    r'D:/Desktop/python_work/MachineLearning/INRIAPerson/original_images/Test/neg.lst')
print(len(pos_list))
print(len(neg_list))
res = detector(pos_list, neg_list)
TPR = res[0] / (res[0] + res[1])
MR = 1 - TPR
FPR = res[2] / (res[2] + res[3])
print("TPR=", TPR)
print("MR =", 1 - TPR)
print("FPR=", FPR)
A = res[0]
data = pd.DataFrame(A)
writer = pd.ExcelWriter('TP.xlsx')  # 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
B = res[1]
data = pd.DataFrame(B)
writer = pd.ExcelWriter('FN.xlsx')  # 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
C = res[2]
data = pd.DataFrame(C)
writer = pd.ExcelWriter('FP.xlsx')  # 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
D = res[3]
data = pd.DataFrame(D)
writer = pd.ExcelWriter('TN.xlsx')  # 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
print(res)
