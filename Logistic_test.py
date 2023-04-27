import numpy as np
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import joblib
import cv2
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt


# 滑动窗口，即得到图片的一部分
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield x, y, image[y: y + window_size[1], x:x + window_size[0]]


def detector(filename):
    im = cv2.imread(filename)
    # 对图片大小进行控制
    im = imutils.resize(im, width=min(400, im.shape[1]))
    # 最小窗口大小
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    # 图像金字塔缩放倍数
    downscale = 1.25
    # 导入训练好的LOG模型
    clf = joblib.load("modelLog")
    # detections存放图片中的人物
    detections = []
    # scale为缩放的次数
    scale = 0
    # 对图片进行缩放
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
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
            pred = clf.predict(fd)
            a = clf.predict_proba(fd)[0, 1]
            b = clf.decision_function(fd)

            # 如果检测到图片中有人
            if pred == 1:
                a = clf.predict_proba(fd)[0,1]
                b = clf.decision_function(fd)
                if clf.decision_function(fd) > 0.5:  ##型预测样本属于positive正样本的可信度
                    # 保存 包含人的矩形
                    detections.append((int(x * (downscale ** scale)), int(y * (downscale ** scale)),
                                       clf.decision_function(fd),
                                       int(min_wdw_sz[0] * (downscale ** scale)),
                                       int(min_wdw_sz[1] * (downscale ** scale))))

        scale += 1

    # 克隆一遍原图，画出矩形，进行输出
    clone = im.copy()

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    # print("shape, ",pick.shape)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 0, 255), 2)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()


img_path = "crop_000016.png"
img = Image.open(img_path)
img.save(img_path)
image = cv2.imread(img_path)
detector(img_path)
