'''          ### 照片__人臉偵測 & 特徵萃取 ###       '''

# 載入所需的套件
import dlib
import cv2
# github上一個很方便的圖像處理包，可以導入python，實現平移，旋轉，調整大小，骨架化等一些操作
import imutils
from imutils import face_utils


# 使用 OpenCV 讀取照片圖檔
img = cv2.imread('/media/hoho/Transcend/備份/專題/CV/dataset_test/pohan.jpg')
img = imutils.resize(img, width=640)     # 縮小圖片，亦可直接使用原圖


# 呼叫Dlib 的人臉偵測器 & 特徵萃取器
'''Dlib 使用的人臉偵測演算法是以方向梯度直方圖（HOG）的特徵加上線性分類器（linear classifier）、
影像金字塔（image pyramid）與滑動窗格（sliding window）來實作。'''
detector = dlib.get_frontal_face_detector()

'''dlib.shape_predictor具有預訓練的地標檢測器，dlib用於估計68個座標(x, y)的位置，這些座標映射在人臉上的面部點
shape_predictor_68_face_landmarks.dat 為訓練好的模型，須先下載存取在本地端
(https://github.com/italojs/facial-landmarks-recognition-/blob/master/shape_predictor_68_face_landmarks.dat)'''
predictor = dlib.shape_predictor(
    '/media/hoho/Transcend/備份/專題/CV/shape_predictor_68_face_landmarks.dat')


# 偵測人臉
'''detector 函數的第二個參數是指定反取樣（unsample）的次數，如果圖片太小的時候，將其設為 1 可讓程式偵較容易測出更多的人臉。'''
face_rects = detector(img, 0)
# print(face_rects)    -> 顯示結果為：rectangles[[(185, 305) (400, 520)]]


# 顯示結果
'''face_rects會回傳兩個tuple，為人臉偵測的方框 ex. [(633, 237), (1079, 683)]'''
for face in face_rects:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # predictor(圖像, 人臉偵測方框座標) 標誌出臉部區域的特徵
    shape = predictor(img, face)

    # 將面部landmarks標誌(x, y)座標轉換成NumPy陣列
    shape = face_utils.shape_to_np(shape)
    # print(shape)   -> 68組(x,y)座標

    # 以方框標示偵測的人臉 (影像, 開始座標, 結束座標, 顏色, 線條寬度<正數為粗細，負數為填滿>, 反鋸齒線條)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)

    for (x, y) in shape:
        # 以圓點標示特徵點 （圖像、圓心、半徑、顏色、線條寬度<正數為粗細，負數為填滿>, 反鋸齒線條）
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)


# 顯示結果 (命名窗口名稱, 欲顯示的圖片)
cv2.imshow('Facial Landmarks', img)


# 確保關閉彈跳的視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
