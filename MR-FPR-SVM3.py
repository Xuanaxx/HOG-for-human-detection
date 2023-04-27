import cv2
import numpy as np
import pandas as pd


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


def detector(pos_list, neg_list):
    winSize = (64, 128)
    blockSize = (16, 16)  # 2*2个cell
    blockStride = (8, 8)  # block移动步长为8个像素（1个cell），在win下105个block
    cellSize = (8, 8)  # cell大小8*8个像素
    nbins = 9  # 每个cell9个bin(方向）
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                            nbins)  # 每个block4个cell，每个cell9个方向，则win下hog特征向量为4*9*105=3780维
    hog.load('myHogDector_set.bin')
    TP = np.zeros([1, 17])
    FN = np.zeros([1, 17])
    FP = np.zeros([1, 17])
    TN = np.zeros([1, 17])
    counter = 0
    for hitThreshold in [k / 10 for k in range(-6, 11, 1)]:
        num_pos = 0
        for i in range(len(neg_list)):
            num_pos += 1
            print(str(num_pos) + '/' + str(len(neg_list)) + '‐' + str(hitThreshold))
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            rects, scores = hog.detectMultiScale(neg_list[i], winStride=(4, 4), padding=(8, 8),
                                                 hitThreshold=hitThreshold,
                                                 scale=1.05)
            if len(rects) > 0:
                FP[0, counter] += 1
            else:
                TN[0, counter] += 1
        num_pos = 0
        for i in range(len(pos_list)):
            num_pos += 1
            print(str(num_pos) + '/' + str(len(pos_list)) + '‐' + str(hitThreshold))
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            rects, scores = hog.detectMultiScale(pos_list[i], winStride=(4, 4), padding=(8, 8),
                                                 hitThreshold=hitThreshold,
                                                 scale=1.05)
            if len(rects) > 0:
                TP[0, counter] += 1
            else:
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
