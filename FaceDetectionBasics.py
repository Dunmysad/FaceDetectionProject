import cv2 as cv
import  mediapipe as mp
import time


# cap = cv.VideoCapture('Video/1-热恋冰激凌- 程Yooooo-1080P 高清-AVC.mp4')
cap = cv.VideoCapture(0)

# 检测模块
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

pTime = 0
while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results) # <class 'mediapipe.python.solution_base.SolutionOutputs'>

    # 显示地标
    if results.detections:
        for id, detection in enumerate(results.detections):
            # 画脸的框
            # mpDraw.draw_detection(img, detection)

            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            # 获取脸位置地标
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img, bbox, (255, 0, 255), 2)
            cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]), (cv.FONT_HERSHEY_PLAIN), 2, (0, 255, 0), 2)




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (20, 70), (cv.FONT_HERSHEY_PLAIN), 3, (255, 0, 0), 2)
    cv.imshow('image', img)
    cv.waitKey(1)