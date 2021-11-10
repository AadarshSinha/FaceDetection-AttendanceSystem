import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#*****************************************FUNCTIONS*********************************
def encode(list):
   temp=[]
   for item in list:
      item=cv2.cvtColor(item,cv2.COLOR_RGB2BGR)
      enc=face_recognition.face_encodings(item)[0]
      temp.append(enc)
   return temp
def attendance(name):
   with open('/Users/aadarsh/Documents/VSCode/FaceDetection/attendance.csv','r+') as file:
       data=file.read
       current_student=[]
       for item in data:
          present=item.split(',')
          current_student.append(present[0])
       if name not in current_student:
          now = datetime.now()
          time  =now.strftime('%H:%M:%S')
          file.writelines(f'\n{name},{time}')
#*******************************************MAIN************************************
path='/Users/aadarsh/Documents/VSCode/FaceDetection/image'
total_student=os.listdir(path)
student_name=[]
student_image=[]

for student in total_student:
       image=cv2.imread(f'{path}/{student}')
       name=os.path.splitext(student)[0]
       student_image.append(image)
       student_name.append(name)

student_image_encode=encode(student_image)

camera=cv2.VideoCapture(0)
print("VIDEO CAPTURE IS ON")
while True:
     success,current_image = camera.read()
     current_image_small=cv2.resize(current_image,(0,0),None,0.25,0.25)
     current_image_small=cv2.cvtColor(current_image_small,cv2.COLOR_RGB2BGR)
     current_image_locations=face_recognition.face_locations(current_image_small)
     current_image_encodings=face_recognition.face_encodings(current_image_small,current_image_locations)

     for face,location in zip(current_image_encodings,current_image_locations):
         found=face_recognition.compare_faces(student_image_encode,face)
         accuracy=face_recognition.face_distance(student_image_encode,face)
         roll_number=np.argmin(accuracy)
         if found[roll_number]:
             name=student_name[roll_number]
             print("Hello ",name)
             y1,x2,y2,x1=location
             y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(current_image,(x1,y1),(x2,y2),(39,138,245),2)
             cv2.rectangle(current_image,(x1,y2-35),(x2,y2),(39,138,245),cv2.FILLED)
             cv2.putText(current_image,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
     cv2.imshow("LIVE",current_image)
     cv2.waitKey(1)
