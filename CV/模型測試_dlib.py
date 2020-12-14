
# 載入所需的套件
import dlib
import math
import cv2
import numpy as np
import glob                             # glob 是文件操作相關的套件
from sklearn.svm import SVC             # Support Vector Classification
import joblib                           # 保存 sklearn 模型的函式庫，亦可使用 pickle


img = ("/media/hoho/Transcend/備份/專題/CV/dataset_test/pohan2.jpg")


# 使用 Dlib 偵測人臉 & 特徵點萃取
detector = dlib.get_frontal_face_detector(
)                                     # 呼叫人臉偵測器
predictor = dlib.shape_predictor(
    "/media/hoho/Transcend/備份/專題/CV/shape_predictor_68_face_landmarks.dat")     # 呼叫特徵點萃取器


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

data = {}
img_data = []


# 定義偵測人臉和特徵點的function
def get_landmarks(img):
    face_rects = detector(img, 0)                   # face_rects 回傳結果為人臉方框的座標
    for k, d in enumerate(face_rects):
        shape = predictor(img, d)                   # 使用predictor類別萃取人臉特徵點

        xlist = []
        ylist = []
        for i in range(1, 68):                      # 將特徵點(共68點) X和Y座標各別存入兩個list
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)                   # 取得 xlist 矩陣的平均值
        ymean = np.mean(ylist)                   # 取得 ylist 矩陣的平均值

        # 每個特徵點的x座標與xmean相減以得到離均差(deviation from the mean)
        xcentral = [(x-xmean) for x in xlist]
        # 每個特徵點的y座標與ymean相減以得到離均差(deviation from the mean)
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        # zip() 函數可以將對應的元素打包成一個個tuple以節約內存
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            # landmarks_vectorised = [xlist[0], ylist[0], xlist[1], ylsit[1]...]
            landmarks_vectorised.append(z)
            # np.asarry() 將input轉為矩陣
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            # linalg=linear algebra，norm則表示範數。範數是對向量（或者矩陣）的度量，是一個標量（scalar）。
            dist = np.linalg.norm(coornp-meannp)        # 計算每個特徵點座標到臉中心點的距離
            landmarks_vectorised.append(dist)
            # math.atan2()回傳從原點到(x, y)點的線段與x軸正方向之間的平面角度(弧度值)
            # 用鼻樑的角度修正臉傾斜所造成的誤差
            landmarks_vectorised.append((math.atan2(y, x)*360)/2*math.pi)

            # 我們得到 a. 每個特徵點到臉中心點的距離  b. 角度  -> 矩陣
            data['landmarks_vectorised'] = landmarks_vectorised

    # 若人臉偵測失敗，value為error
    if len(face_rects) < 1:
        data['landmarks_vectorised'] = "error"


def test(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    get_landmarks(clahe_image)
    img_data.append(data['landmarks_vectorised'])
    np_img_data = np.array(img_data)
    return img_data


# 呼叫模型
clf_1 = joblib.load('/media/hoho/Transcend/備份/專題/CV/clf_5.pkl')


# 套用模型
print(clf_1.predict(test(img)))
