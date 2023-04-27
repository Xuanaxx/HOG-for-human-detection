import cv2
import numpy as np
import random


# 对输入图进行灰度处理和gamma矫正
def adjust_gamma(img, gamma=1.0):  # img已灰度化
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


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


# wsize: 处理图片大小，通常64*128(64列，128行）; 输入图片尺寸>= wsize
def computeHOGs(img_lst, gradient_lst, wsize=(128, 64)):
    winSize = (64, 128)
    blockSize = (16, 16)  # 2*2个cell
    blockStride = (8, 8)  # block移动步长为8个像素（1个cell），在win下105个block
    cellSize = (8, 8)  # cell大小8*8个像素
    nbins = 9  # 每个cell9个bin(方向）
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                            nbins)  # 每个block4个cell，每个cell9个方向，则win下hog特征向量为4*9*105=3780维
    # hog.winSize = wsize
    for i in range(len(img_lst)):
        if img_lst[i].shape[1] >= wsize[1] and img_lst[i].shape[0] >= wsize[0]:  # shape[0]=160
            roi = img_lst[i][(img_lst[i].shape[0] - wsize[0]) // 2: (img_lst[i].shape[0] - wsize[0]) // 2 + wsize[0], \
                  (img_lst[i].shape[1] - wsize[1]) // 2: (img_lst[i].shape[1] - wsize[1]) // 2 + wsize[1]]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
            gradient_lst.append(hog.compute(
                gray))  # hog.compute(gray)为3780*1维向量,gradient_lst为3780*（2416+12180）维数组（正样本2416张，负样本crop1218*10张）
    # return gradient_lst


def get_svm_detector(svm):
    '''
    导出可以用于cv2.HOGDescriptor()的SVM检测器，实质上是训练好的SVM的支持向量和rho参数组成的列表
    :param svm: 训练好的SVM分类器
    :return: SVM的支持向量和rho参数组成的列表，可用作cv2.HOGDescriptor()的SVM检测器
    '''
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


# 主程序
# 第一步：计算HOG特征
neg_list = []
pos_list = []
gradient_lst = []
labels = []
hard_neg_list = []

pos_list = load_images(r'D:/Desktop/python_work/MachineLearning/INRIAPerson/normalized_images/train/pos.lst')
full_neg_lst = load_images(
    r'D:/Desktop/python_work/MachineLearning/INRIAPerson/normalized_images/train/neg.lst')
sample_neg(full_neg_lst, neg_list, [128, 64])
print(len(neg_list))
computeHOGs(pos_list, gradient_lst)
[labels.append(+1) for _ in range(len(pos_list))]  # pos_list=2416
computeHOGs(neg_list, gradient_lst)
[labels.append(-1) for _ in range(len(neg_list))]  # neg_list=12180（每张neg图片crop10，有1218张图片）,label为14596

# 第二步：训练SVM
svm = cv2.ml.SVM_create()
svm.setCoef0(0)
svm.setCoef0(0.0)
svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
svm.setTermCriteria(criteria)
svm.setGamma(0)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
svm.setC(0.01)  # From paper, soft classifier
svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

# 第三步：加入识别错误的样本，进行第二轮训练
# 参考 http://masikkk.com/article/SVM-HOG-HardExample/
winSize = (64, 128)
blockSize = (16, 16)  # 2*2个cell
blockStride = (8, 8)  # win下105个blcok
cellSize = (8, 8)  # cell大小8*8个像素
nbins = 9  # 每个cell9个bin
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)  # 3780
hard_neg_list.clear()
hog.setSVMDetector(get_svm_detector(svm))
for i in range(len(full_neg_lst)):
    rects, wei = hog.detectMultiScale(full_neg_lst[i], winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        hardExample = full_neg_lst[i][y:y + h, x:x + w]
        hard_neg_list.append(cv2.resize(hardExample, (64, 128)))
computeHOGs(hard_neg_list, gradient_lst)
[labels.append(-1) for _ in range(len(hard_neg_list))]
svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

# 第四步：保存训练结果
hog.setSVMDetector(get_svm_detector(svm))
hog.save('myHogDector_set.bin')
