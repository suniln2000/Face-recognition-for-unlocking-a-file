from base64 import encode
import os
import cv2
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import face_recognition

path='/home/a/Desktop/computer vision/project/face detect project/images'
#for storing images and images names
images=[]
classnames=[]

mylist=os.listdir(path)

for cl in mylist:
    currentimage=cv2.imread(f'{path}/{cl}') #path for read each images
    images.append(currentimage)
    classnames.append(os.path.splitext(cl)[0])  #to split the image name

print(classnames)

def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

encodelistKnownfaces=findencodings(images)
print(encodelistKnownfaces)


# initialize a flag for whether a match has been found
match_found = False

cap = cv2.VideoCapture(0)

# define other variables and functions

while True:
    success, img = cap.read()


    while not match_found:
        success,img=cap.read()
        imgsmall=cv2.resize(img,(0,0),None,0.25,0.25) #to resize the image quality from webcam
        faceinframe=face_recognition.face_locations(imgsmall)

        faceencode=face_recognition.face_encodings(imgsmall,faceinframe)

        for faceencode,faceloc in zip(faceencode,faceinframe):
            matches=face_recognition.compare_faces(encodelistKnownfaces,faceencode)

            facedistance=face_recognition.face_distance(encodelistKnownfaces,faceencode)

            matchindex=np.argmin(facedistance)

            #to print the name
            if matches[matchindex]:
                name = classnames[matchindex]

                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                cv2.rectangle(img, (x1, y1), (x2, y2), (225, 0, 0), 2)
                cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (225, 0, 0), 2)

                # show a message that the face matched
                cv2.putText(img, 'Matched', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # show a countdown for 3 seconds
                for i in range(3, 0, -1):
                    cv2.putText(img, str(i), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('face recognition', img)
                    cv2.waitKey(1000)
                    img = cap.read()[1]

                # open the file
                path = '/home/a/Desktop/face.txt'
                if matchindex == 0:
                    os.system('xdg-open ' + path)

                # set the flag to indicate that a match has been found
                match_found = True
                break

            else:
                y1,x2,y2,x1=faceloc
                y1,x2,y2,x1=y1*3,x2*3,y2*3,x1*3
                cv2.putText(img,'not matched',(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        cv2.imshow('face recongnition',img)
        if cv2.waitKey(1) & 0XFF==27:
            break

    # release the camera
    cap.release()

    # close all windows
    cv2.destroyAllWindows()

