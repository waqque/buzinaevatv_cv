import cv2
import numpy as np
import json
import os
import random

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
capture.set(cv2.CAP_PROP_EXPOSURE,-5)
cv2.namedWindow("Mask",cv2.WINDOW_NORMAL)

color = (56, 160, 100) 
lower = np.array([50, 130, 80])
upper = np.array([62, 255, 255])

def getColor(image):
    x,y,w,h=cv2.selectROI("Color selection",image)
    roi=image[y:y+h,x:x+w]
    color =(np.median(roi[:, :, 0]),
            np.median(roi[:, :, 1]),
            np.median(roi[:, :, 2]))
    cv2.destroyWindow('Color selection')
    
    return color

def getBall(image,color):
    lower=(np.max(color[0]-5,0),color[1]*0.8,color[2]*0.8)
    upper=(color[0]+5,255,255)
    mask=cv2.inRange(image,lower,upper)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>0:
        contour=max(contours,key=cv2.contourArea)
        (x,y),radius=cv2.minEnclosingCircle(contour)
        return True,(int(x),int(y),int(radius),mask)
    return False,(-1,-1,-1,np.array([]))

path="setting.json"
if os.path.exists(path):
    base_colors=json.load(open(path,"r"))
else:
    base_colors={}
game_started=False
guess_colors=[]
while capture.isOpened():
    ret, frame = capture.read()
    blurred=cv2.GaussianBlur(frame,(11,11),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q':
        break
    if key in ("1", "2", "3"): #1 - green, 2- blue, 3-orange
        color=getColor(hsv)
        base_colors[key]=color
        print(base_colors)
    player=[]
    for key in base_colors:
        temp=False
        retr,(x,y,radius,mask)=getBall(hsv,base_colors[key])
        if retr:
            cv2.imshow("Mask",mask)
            cv2.circle(frame,(x,y),radius,(0, 0, 0),3)
            player.append((key,x))
            if len(player)==3:
                break
    player=sorted(player,key=lambda x:x[1])
    if len(base_colors)==3:
        if not game_started:
            guess_colors=list(base_colors)
            random.shuffle(guess_colors)
            game_started=True
            print(guess_colors)
    if game_started:
        t=list(x[0] for x in player)
        if t==guess_colors:
            cv2.putText(frame,f"Molodets WEll done",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0))
    cv2.putText(frame,f"Game started = {game_started}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0))
    cv2.imshow('Camera', frame)

capture.release()
cv2.destroyAllWindows()

json.dump(base_colors,open(path,"w"))