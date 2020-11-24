'''         Emotion Recognition using Facial Landmarks, Python, DLib and OpenCV         '''

# 載入所需的套件
import dlib
import cv2
import numpy as np
import glob             # glob 是文件操作相關的套件
import random
import math
import itertools
from sklearn.svm import SVC             # Support Vector Classification
import joblib    # 保存 sklearn 模型的函式庫，亦可使用 pickle


# 定義情緒種類，dataset以以下資料夾分類
emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# 定義字適應
'''Contrast Limited Adaptive Histogram Equalization（自適應值方圖均衡化），是一種計算機圖像處理技術，用於提高圖像的對比度
clipLimit參數表示對比度的大小，tileGridSize參數表示每次處理塊的大小 。'''
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# 使用 Dlib 偵測人臉 & 特徵點萃取
detector = dlib.get_frontal_face_detector(
)                                     # 呼叫人臉偵測器
predictor = dlib.shape_predictor(
    "./shape_predictor_68_face_landmarks.dat")     # 呼叫特徵點萃取器


'''    ★  SVM w/ Linear Kernel ★ （以下kernel參數可調整再訓練）  '''

'''kernel：參數選擇有RBF(高斯核), Linear(線性核函數)，Poly(多項式函數), Sigmoid(sigmoid核函數)
   probability：是否啟用概率估計， tol：停止訓練的誤差值大小， '''
clf = SVC(kernel='linear', probability=True, tol=1e-5, verbose=False)


data = {}           # 將所有特徵點的值帶入字典   data['landmarks_vectorised'] = []


# 定義一個function蒐集並建立所有圖片檔案名稱的list, 隨機將資料分成80/20
def get_files(emotion):
    files = glob.glob("./dataset_test/2_SAVEE/%s/*" %
                      emotion)      # 將所有檔案名稱回傳並轉為list型別
    random.shuffle(files)                           # 將list裡面的檔案名稱順序洗牌
    training = files[:int(len(files)*0.8)]          # 取得list前面80%的檔案做為訓練集
    prediction = files[-int(len(files)*0.2):]       # 取得list後面20%的檔案做為測試集
    return training, prediction


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


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    # 在每個emotion資料夾做迴圈
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)

        # 將數據附加到訓練集的list和測試集的list中，並生成標籤0〜7
        for item in training:                   # 對訓練集的每一張照片做迴圈
            img = cv2.imread(item)              # 讀取訓練集的照片
            # 轉換為灰階照片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)          # 取得特徵點的矩陣

            # 若value為error， print出此照片沒有偵測到人臉
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")

            else:
                # 若有偵測道人臉，將特徵點矩陣加入training_data
                training_data.append(data['landmarks_vectorised'])
                # 將對應的情緒標籤號碼加入training_labels
                training_labels.append(emotions.index(emotion))

        for item in prediction:                 # 對測試集的每一張照片做迴圈
            img = cv2.imread(item)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)

            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


accuracy_lin = []

# 將模型訓練十次
for i in range(0, 10):
    print("Making sets %s" % i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    # 將訓練集轉成numpy矩陣以放進分類器進行學習
    npar_train = np.array(training_data)
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" % i)
    clf.fit(npar_train, training_labels)     # 將訓練集帶進SVM進行訓練

    # 使用score()函數取得準確率
    print("getting accuracies %s" % i)
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print("linear: ", pred_lin)
    accuracy_lin.append(pred_lin)               # 將準確率加入accur_lin


# 取得訓練十次的準確率
print("Mean value lin svm: %s" % np.mean(accuracy_lin))


# 儲存訓練好的模型
'''前面的clf為我們上面的svm.SVC()，後面的clf.pkl為檔名'''
joblib.dump(clf, 'clf_1.pkl')
