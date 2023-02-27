from AIKoreaHand import AIKoreaHands
import cv2
import time
from overlays import overlay_transparent
import mediapipe as mp

cap = cv2.VideoCapture(0)


success, frame = cap.read()
hh = frame.shape[0]
wh = frame.shape[1]


print(hh,wh)


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


#--------------720x1280----------------------
# bgArray = [
#     cv2.imread(r"E:\ing\aikorea\2D\bgi\bgi0.png", cv2.IMREAD_UNCHANGED),
#     cv2.imread(r"E:\ing\aikorea\2D\bgi\bgi1.png", cv2.IMREAD_UNCHANGED),
#     cv2.imread(r"E:\ing\aikorea\2D\bgi\bgi2.png", cv2.IMREAD_UNCHANGED),
#     cv2.imread(r"E:\ing\aikorea\2D\bgi\bgi3.png", cv2.IMREAD_UNCHANGED),
#     cv2.imread(r"E:\ing\aikorea\2D\bgi\bgi4.png", cv2.IMREAD_UNCHANGED),
#     cv2.imread(r"E:\ing\aikorea\2D\bgi\bgi5.png", cv2.IMREAD_UNCHANGED)
# ]

#--------------1080x1920----------------------
bgArray = [
    cv2.imread(r"E:\ing\aikorea\2D\bgi_new\bgi0.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"E:\ing\aikorea\2D\bgi_new\bgi1.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"E:\ing\aikorea\2D\bgi_new\bgi2.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"E:\ing\aikorea\2D\bgi_new\bgi3.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"E:\ing\aikorea\2D\bgi_new\bgi4.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"E:\ing\aikorea\2D\bgi_new\bgi5.png", cv2.IMREAD_UNCHANGED)
]

for i in range(0, len(bgArray)):
    bgArray[i] = cv2.resize(bgArray[i], (wh,hh))

pTime=0

teeth1 = cv2.imread(r"E:\ing\aikorea\2D\Image for Dee\ff1.png", -1)
teeth2 = cv2.imread(r"E:\ing\aikorea\2D\Image for Dee\ff2.png", -1)
teeth3 = cv2.imread(r"E:\ing\aikorea\2D\Image for Dee\ff3.png", -1)
teeth4 = cv2.imread(r"E:\ing\aikorea\2D\Image for Dee\ff4.png", -1)

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

while True:
    success, frame = cap.read()

    frame, fingers = AIKoreaHands(frame)
    image = frame.copy()
    print(fingers)

    #hand detector
    frame.flags.writeable=True
    font = cv2.FONT_HERSHEY_SIMPLEX

    #face detector
    #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_detection.process(image)

    # 영상에 얼굴 감지 주석 그리기 기본값 : True.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if fingers != "none":
        if results.detections:
            for detection in results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                xmin = int(relative_bounding_box.xmin * wh)-50
                ymin = int(relative_bounding_box.ymin * hh)-80

                top_left = (xmin,ymin)
                width = int(relative_bounding_box.width * wh) + 100
                height = int(relative_bounding_box.height * hh) + 100

                #face detector output
                mp_drawing.draw_detection(image, detection)

                if(fingers=="one"):
                    teethimage = cv2.resize(teeth1, (width, height),interpolation=cv2.INTER_CUBIC)
                    overlay_transparent(frame, teethimage, top_left[0], top_left[1])
                    overlay_transparent(frame, bgArray[1], 0,0)
                elif(fingers=="two"):
                    teethimage = cv2.resize(teeth2, (width, height),interpolation=cv2.INTER_CUBIC)
                    overlay_transparent(frame, teethimage, top_left[0], top_left[1])
                    overlay_transparent(frame, bgArray[2], 0,0)
                elif(fingers=="three"):
                    teethimage = cv2.resize(teeth3, (width, height),interpolation=cv2.INTER_CUBIC)
                    overlay_transparent(frame, teethimage, top_left[0], top_left[1])
                    overlay_transparent(frame, bgArray[3], 0,0)
                elif(fingers=="four"):
                    teethimage = cv2.resize(teeth4, (width, height),interpolation=cv2.INTER_CUBIC)
                    overlay_transparent(frame, teethimage, top_left[0], top_left[1])
                    overlay_transparent(frame, bgArray[4], 0,0)
                elif(fingers=="five"):
                    overlay_transparent(frame, bgArray[5], 0,0)
    else:
        overlay_transparent(frame, bgArray[0], 0,0)

    cTime = time.time()
    sec = cTime - pTime
    fps = 1 / (sec)
    pTime = cTime

    print(int(fps))

    cv2.putText(frame, f'FPS: {int(fps)}',(400,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),3)

    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WND_PROP_FULLSCREEN)
    cv2.imshow("image", frame)
    resize_frame = cv2.resize(frame, dsize=(360, 640), interpolation=cv2.INTER_LINEAR)  # 키울 그림, dsize - 자신의 해상도 결정
    cv2.imshow('cam', resize_frame)

    if cv2.waitKey(1) == ord('q'):
        break
