
# import libraries
import cv2
import numpy as np

# import the cascades for human faces and full body
faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
bodyCascade = cv2.CascadeClassifier("Resources/fullbody_recognition_model.xml")

# import the dataset
cap = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480

while 1 :
    # import the dataset
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))

    # training the model
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
    full_body = bodyCascade.detectMultiScale(imgGray,1.1,4)
    objectType1 = "face"
    objectType2 = "human"

    # testing the model(human detection and recognation)
    for (x,y,w,h) in full_body:

        # drawing the rectangle on fullbody human
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(img, objectType2,
                    (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                    (255, 255, 255), 2)
        # drawing the rectangle on human face
    for (x1, y1, w1, h1) in faces:
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.putText(img, objectType1,
                    (x1, y1 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.9,
                    (255, 255, 255), 2)

   # visualsation the model
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



