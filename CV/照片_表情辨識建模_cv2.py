'''         Emotion Recognition using Facial Landmarks, Python and OpenCV         '''

# 載入所需的套件
import numpy as np
import cv2
import glob
import random
import numpy as np


# 呼叫 OpenCV 的人臉偵測器，共四種，皆使用Haar演算法
faceDet = cv2.CascadeClassifier(
    "./OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(
    "./OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier(
    "./OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier(
    "./OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")


# 定義情緒種類
emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def detect_faces(emotion):
    files = glob.glob("Dataset/1_kaggle/images/images/train/%s/*" %
                      emotion)      # 將所有檔案名稱回傳並轉為list型別
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)       # 讀取照片
        # 轉換為灰階照片
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 定義四種分類器偵測人臉
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(
            5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        # 逐一使用四種分類器， 若已偵測到人臉即停止行為，若沒有偵測到人臉則回傳空值
        if len(face) == 1:
            face_rects = face
        elif len(face_two) == 1:
            face_rects = face_two
        elif len(face_three) == 1:
            face_rects = face_three
        elif len(face_four) == 1:
            face_rects = face_four
        else:
            face_rects = ""

        # 裁剪並儲存人臉
        for (x, y, w, h) in face_rects:  # 取得人臉方框座標(x1, y1) 、(x2, y2)
            print("face found in file: %s" % f)
            gray = gray[y:y+h, x:x+w]  # Cut the frame to size
            try:
                # 將全部照片尺寸調整至相同大小
                out = cv2.resize(gray, (350, 350))
                cv2.imwrite("dataset_test/1_Kaggle/%s/%s.jpg" %
                            (emotion, filenumber), out)  # 寫入照片
            except:
                pass     # 若產生錯誤則跳過該照片
        filenumber += 1  # 繼續偵測下一張照片


for emotion in emotions:
    detect_faces(emotion)  # 在全部情緒資料夾內呼叫人臉偵測的function


'''         Creating the training and Classification Set            '''
emotions = ["neutral", "anger", "disgust", "fear",
            "happy", "sadness", "surprise"]  # Emotion list

# 呼叫 FisherFace 分類器
fishface = cv2.face.FisherFaceRecognizer_create()
data = {}


# 定義一個function蒐集並建立所有圖片檔案名稱的list, 隨機將資料分成80/20
def get_files(emotion):
    files = glob.glob("Dataset/train/%s/*" % emotion)    # 將所有檔案名稱回傳並轉為list型別
    random.shuffle(files)                                # 將list裡面的檔案名稱順序洗牌
    training = files[:int(len(files)*0.8)]               # 取得list前面80%的檔案做為訓練集
    prediction = files[-int(len(files)*0.2):]            # 取得list後面20%的檔案做為測試集
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)

        # 將數據附加到訓練集的list和測試集的list中，並生成標籤0〜7
        for item in training:
            image = cv2.imread(item)  # 讀取照片
            # 轉換為灰階照片
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 若有偵測到人臉，將影像矩陣加入training_data
            training_data.append(gray)
            # 將對應的情緒標籤號碼加入training_labels
            training_labels.append(emotions.index(emotion))

        for item in prediction:  # 對測試集的每一張照片重複以上步驟
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))


# 將模型訓練十次
metascore = []
for i in range(0, 10):
    correct = run_recognizer()
    print("got", correct, "percent correct!")
    metascore.append(correct)
print("\n\nend score:", np.mean(metascore), "percent correct!")
