import cv2
vidcap = cv2.VideoCapture(0,cv2.CAP_V4L)
CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 1200
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    #m=0
    if hasFrames:
        #print(vidcap.get(cv2.CAP_PROP_FPS))
        cv2.imwrite("capture/left/image"+str(count)+'time-'+str(sec)+".png", image)     # save frame as PNG file
    return hasFrames
sec = 0
frameRate = 0.1 #//it will try capture image in each 0.1 second
m=0
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
