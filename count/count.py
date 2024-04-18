import cv2


# 读取图片并转换为灰度图
img_path = 'data/balls.png'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 均值滤波核大小
blur_size = 21
# 获取当前的二值化阈值
threshold = 188
# 获取当前的腐蚀核大小
erode = 5
# 获取当前的膨胀核大小
dilate = 12
# 应用均值滤波
blur = cv2.blur(gray, (blur_size, blur_size))
# 应用二值化
_, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
# 应用膨胀腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
binary = cv2.erode(binary, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
binary = cv2.dilate(binary, kernel)
num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary)

print(f'球的数量为：{num_labels - 1}')