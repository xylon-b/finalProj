import numpy as np
import cv2
from matplotlib import pyplot as plt



imgL = cv2.imread("./data/view0.png", cv2.IMREAD_COLOR)
imgR = cv2.imread("./data/view5.png", cv2.IMREAD_COLOR)

# 이미지 크기 조정
resized_width = 640
resized_height = 480
imgL_resized = cv2.resize(imgL, (resized_width, resized_height))
imgR_resized = cv2.resize(imgR, (resized_width, resized_height))

# 스테레오 매칭 알고리즘 변경 (SGBM)
window_size = 3
min_disp = 0
num_disp = 64 + 16 + 16
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 스테레오 매칭 수행
disparity = stereo.compute(imgL_resized, imgR_resized)

plt.imshow(disparity,cmap='gray')
plt.show()
