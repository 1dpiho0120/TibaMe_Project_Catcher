'''          ### 影片__人臉偵測 & 特徵萃取 ###       '''

# 載入所需的套件
import dlib
import cv2
# github上一個很方便的圖像處理包，可以導入python，實現平移，旋轉，調整大小，骨架化等一些操作
import imutils
from imutils import face_utils      # 人臉特徵萃取套件


# 使用 OpenCV 讀取影片檔案
cap = cv2.VideoCapture(
    '/media/hoho/Transcend/備份/專題/CV/dataset_test/people.mp4')


# 取得畫面尺寸
'''
通過cap.get(propID)訪問影片的某些功能，propID是0到18之間的數字，每個數字分別表示影片的屬性，例如：
cv2.CAP_PROP_FRAME_WIDTH -> propID=3 | cv2.CAP_PROP_FRAME_HEIGHT -> propID=4
分別可得到影片的寬度和高度。如果想修改為320x240, 只須在後面加入欲修改成的寬x高：
width = cap.set(cv2.CAP_PROP_FRAME_WIDTH，320) | height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
'''
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# 定義視訊編解碼器的型別
'''
VideoWriter_fourcc為視訊編解碼器。視訊編解碼器是指一個能夠對數位影片進行壓縮或者解壓縮的程式或者裝置。
cv2.VideoWriter_fourcc(‘X’, ‘V’, ‘I’, ‘D’),該引數是MPEG-4編碼型別，
以字尾*.avi, *.mp4等結尾的檔案有一部分是使用這些視訊編解碼器的，這邊使用 XVID 編碼
'''
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# 建立 VideoWriter 物件，輸出影片至 output.mp4，FPS 值為 20.0
'''
cv2.VideoWriter() -> 使用這個函數可以創建一個用於寫出影片文件的句柄
第一個參數是指定輸出的檔名，例如：下列範例中的 output.mp4，
第二個參數為指定 FourCC，即為先前第33行設定寫出影片的編碼格式
第三個參數為 fps 影像偵率，FPS值 = 影格率，是用於測量顯示影格數的量度。一般來說FPS用於描述影片、電子繪圖或遊戲每秒播放多少影格
第四個參數為先前取得的原影像大小
'''
out = cv2.VideoWriter(
    '/media/hoho/Transcend/備份/專題/CV/dataset_test/output.mp4', fourcc, 20.0, (width, height))


# Dlib 的人臉偵測器 & 特徵萃取器
'''Dlib 使用的人臉偵測演算法是以方向梯度直方圖（HOG）的特徵加上線性分類器（linear classifier）、
影像金字塔（image pyramid）與滑動窗格（sliding window）來實作。'''
detector = dlib.get_frontal_face_detector()

'''
shape_predictor_68_face_landmarks.dat 為訓練好的模型，須先下載存取在本地端
(https://github.com/italojs/facial-landmarks-recognition-/blob/master/shape_predictor_68_face_landmarks.dat)
'''
predictor = dlib.shape_predictor(
    "/media/hoho/Transcend/備份/專題/CV/shape_predictor_68_face_landmarks.dat")


# 以迴圈從影片檔案讀取影格，並顯示出來
'''
在這個無窮迴圈中，每次呼叫 cap.read() 就會讀取一張畫面，
其第一個傳回值 ret代表讀取成功與否（True 代表成功，False 代表失敗），而第二個傳回值 frame 就是影片的單張畫面。
'''
while True:
    ret, frame = cap.read()
    # print(ret)

    '''
    這裡我們改用 detector.run 來偵測人臉，它的第三個參數是指定分數的門檻值，
    所有分數超過這個門檻值的偵測結果都會被輸出，而傳回的結果除了人臉的位置(face_rects)之外，演算的結果會有一個分數(scores)與子偵測器的編號(idx), 
    此分數愈大, 表示愈接近人臉，分數愈低表示愈接近誤判。子偵測器的編號則可以用來判斷人臉的方向。
    '''
    # 偵測人臉
    face_rects, scores, idx = detector.run(frame, 1)
    #print(scores, idx)

    # 取出所有偵測的結果
    '''face_rects會回傳兩個tuple，為人臉偵測的方框 ex. [(633, 237), (1079, 683)]'''
    for face in face_rects:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        shape = predictor(frame, face)             # 在每一個單張畫面的人臉方框檢查特徵點
        shape = face_utils.shape_to_np(shape)      # 把臉部特徵點座標轉化為數組 Numpy array

        # 以方框標示偵測的人臉 (影像, 開始座標, 結束座標, 顏色, 線條寬度<正數為粗細，負數為填滿>, 反鋸齒線條)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        for (x, y) in shape:
            # 以圓點標示特徵點 （圖像、圓心、半徑、顏色、線條寬度<正數為粗細，負數為填滿>, 反鋸齒線條）
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)

    # 顯示結果 (命名窗口名稱, 欲顯示的圖片)
    video = cv2.imshow("Facial Landmarks", frame)

    '''
    如下面的判斷式——若使用者没有按下q键,就會持續等待(循環)，直到觸發後執行break跳出迴圈
    0xFF是十六進制常數，二進制值為11111111，和後面的ASCII碼對照。ord(' ')可以將字符轉化為對應的整數(ASCII碼)
    此動作是為了防止BUG
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 確保關閉彈跳的視窗
cv2.destroyAllWindows("Facial Landmarks")
