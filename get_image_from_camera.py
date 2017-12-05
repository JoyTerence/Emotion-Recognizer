import numpy as np
import cv2


current_dir = r"C:\\Users\\Joy.DESKTOP-M53NCFS\\Documents\\GitHub\\Emotion-recognizer"
face_cascade = cv2.CascadeClassifier(current_dir + "\\haarcascades\\haarcascade_frontalface_default.xml")




def get_image():
    cap = cv2.VideoCapture(0)
    got_face = False
    while 1:
        ret, img = cap.read()
        cv2.imshow('img',img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print ("end")
            break

        for (x,y,w,h) in faces:
            print ("face detected")
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            face_roi = img[y:y+h,x:x+w]
            resized_face = cv2.resize(face_roi, (48, 48))
            gray_resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            got_face = True





    if got_face:
        print ("got face")
        cv2.imwrite(current_dir+"\\images\\img.jpg", gray_resized_face)
        cap.release()
        cv2.destroyAllWindows()
        return gray_resized_face
