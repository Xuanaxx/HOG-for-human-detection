import cv2
from PIL import Image

winSize = (64, 128)
blockSize = (16, 16)  # 2*2个cell
blockStride = (8, 8)  # block移动步长为8个像素（1个cell），在win下105个block
cellSize = (8, 8)  # cell大小8*8个像素
nbins = 9  # 每个cell9个bin(方向）
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                        nbins)  # 每个block4个cell，每个cell9个方向，则win下hog特征向量为4*9*105=3780维
hog.load('myHogDector_set.bin')

# 解决png文件打开报错问题
img_path = "crop001501.png"
img = Image.open(img_path)
img.save(img_path)

img = cv2.imread("crop001501.png")
rects, wei = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8),scale=1.05)
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('a', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
