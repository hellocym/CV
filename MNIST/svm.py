import cv2
import numpy as np
from skimage.feature import hog, haar_like_feature
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载MNIST数据集
digits = datasets.load_digits()
X_digits = digits.images
y_digits = digits.target

# 提取HOG特征
def extract_hog_features(images):
    hog_features = []
    for image in images:
        fd = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)


# 提取Haar特征
def extract_haar_features(images):
    haar_features = []
    for image in images:
        feature = haar_like_feature(image, 0, 0, image.shape[0], image.shape[1], 'type-2-x')
        haar_features.append(feature)
    return np.array(haar_features)


# 提取特征
hog_features = extract_hog_features(X_digits)
# lbp_features = extract_lbp_features(X_digits)
haar_features = extract_haar_features(X_digits)

# 组合HOG和Haar特征
combined_features = np.hstack((hog_features, haar_features))

# 划分数据集
X_train_hog, X_test_hog, y_train, y_test = train_test_split(hog_features, y_digits, test_size=0.2, random_state=42)
# X_train_lbp, X_test_lbp = train_test_split(lbp_features, test_size=0.2, random_state=42)
X_train_haar, X_test_haar = train_test_split(haar_features, test_size=0.2, random_state=42)
X_train_combined, X_test_combined = train_test_split(combined_features, test_size=0.2, random_state=42)

# 创建SVM分类器
svm_hog = SVC()
# svm_lbp = SVC()
svm_haar = SVC()
svm_combined = SVC()

# 训练分类器
svm_hog.fit(X_train_hog, y_train)
# svm_lbp.fit(X_train_lbp, y_train)
svm_haar.fit(X_train_haar, y_train)
svm_combined.fit(X_train_combined, y_train)

# 测试分类器
y_pred_hog = svm_hog.predict(X_test_hog)
# y_pred_lbp = svm_lbp.predict(X_test_lbp)
y_pred_haar = svm_haar.predict(X_test_haar)
y_pred_combined = svm_combined.predict(X_test_combined)

# 输出评估结果
print("HOG特征分类结果:")
print(classification_report(y_test, y_pred_hog))
# print("LBP特征分类结果:")
# print(classification_report(y_test, y_pred_lbp))
print("Haar特征分类结果:")
print(classification_report(y_test, y_pred_haar))
print("HOG+Haar特征分类结果:")
print(classification_report(y_test, y_pred_combined))