
import numpy as np
import cv2
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize

# Filtering
kernel= np.ones((3,3),np.uint8)

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')
        
# This section has to be uncommented if you want to take mesurements and store them in the excel
##        ws.append([counterdist, average])
##        print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
##        if (counterdist <= 85):
##            counterdist += 3
##        elif(counterdist <= 120):
##            counterdist += 5
##        else:
##            counterdist += 10
##        print('Next distance to measure: '+str(counterdist)+'cm')

# Mouseclick callback
wb=Workbook()
ws=wb.active  

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Prepare object points
objp = np.zeros((4*5,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(5,280):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    j=5
    t= str(i)
    ChessImaR= cv2.imread('WorkPlace/20_05_08_newdata/chessboard_two/right/'+t+'.png')    # Right side
    ChessImaL= cv2.imread('WorkPlace/20_05_08_newdata/chessboard_two/left/'+t+'.png')    # Left side
    grayR = cv2.cvtColor(ChessImaR,cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(ChessImaL,cv2.COLOR_BGR2GRAY)
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (4,5),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (4,5),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(grayR,cornersR,(10,10),(-1,-1),criteria)
        cv2.cornerSubPix(grayL,cornersL,(10,10),(-1,-1),criteria)
        #show two camera chessboard matching image 
        #imgR = cv2.drawChessboardCorners(ChessImaR, (4,5), cornersR,retR)
        #cv2.putText(imgR, str(i), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        #cv2.imwrite('selected_chessboard_image_two_camera/right/'+t+'.png',imgR)
        #key = cv2.waitKey(1500)
        #imgL = cv2.drawChessboardCorners(ChessImaL, (4,5), cornersL,retL)
        #cv2.putText(imgL, str(i), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        #cv2.imwrite('selected_chessboard_image_two_camera/left/'+t+'.png',imgL)
        #key = cv2.waitKey(1000)
        #######
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)
    j+=10
# Determine the new values for different parameters
N_OK = len(objpoints)
print("Found " + str(N_OK) + " valid images for calibration")
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        grayR.shape[::-1],None,None)
hR,wR= grayR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))
#print(mtxR,distR)
#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        grayL.shape[::-1],None,None)
#print(mtxL,distL)
hL,wL= grayL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))
print('optimal\n')
print(OmtxL)
balance=0.0
dim2=None
dim3=None
DIM=(1600, 1200)
dim1 = (1600,1200)  #dim1 is the dimension of input image to un-distort
print(dim1)
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1
scaled_K1 = mtxR * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K2 = mtxL * dim1[0] / DIM[0]
scaled_K1[2][2] = 1.0  # Except that K[2][2] is always 1.0
scaled_K2[2][2] = 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
#mtxR, rev = cv2.getOptimalNewCameraMatrix(scaled_K1,distR,(wR,hR),0.00,(wR,hR))
#mtxL,rev = cv2.getOptimalNewCameraMatrix(scaled_K2,distL,(wL,hL),0.00,(wL,hL))
print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          grayR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

print(MLS, dLS, MRS, dRS)
# StereoRectify function
rectify_scale= 1 # if 0 image croped, if 1 image not croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 grayR.shape[::-1], R, T,flags,rectify_scale)  # last paramater is alpha, if 0= croped, if 1= not croped
print('\n',RL, RR, PL, PR,Q)
print('Perspective Matrix\n',Q)
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             grayL.shape[::-1],  cv2.CV_32FC1)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              grayR.shape[::-1],  cv2.CV_32FC1)

frameL=cv2.imread('WorkPlace/20_05_08_newdata/chessboard_two/80_left.png')
frameR=cv2.imread('WorkPlace/20_05_08_newdata/chessboard_two/80_right.png')
Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LINEAR)  # Rectify the image using the kalibration parameters founds during the initialisation
Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LINEAR)
#for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
#     Left_nice[line*20,:]= (0,0,255)
#     Right_nice[line*20,:]= (0,0,255)
frameL=cv2.imwrite('stereo_images/left/left_rectify_for_3d.png',Left_nice)
frameR=cv2.imwrite('stereo_images/right/right_rectify_for_3d.png',Right_nice)


# upload alreday distorted and rectified images
l = np.array([[ 1002],[ 760]],dtype=np.float)
r = np.array([[ 871 ],[ 671]],dtype=np.float)
#points = cv2.triangulatePoints(PL,PR,ptl,ptr)
print (points)
#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras
CamR= cv2.VideoCapture(0)   # Wenn 0 then Right Cam and wenn 2 Left Cam
CamL= cv2.VideoCapture(1)

while True:
    # Start Reading Camera images
    #retR, frameR= CamR.read()
    #retL, frameL= CamL.read()

    # Rectify the images on rotation and alignement
    Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
    Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

##    # Draw Red lines
##    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
##        Left_nice[line*20,:]= (0,0,255)
##        Right_nice[line*20,:]= (0,0,255)
##
##    for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
##        frameL[line*20,:]= (0,255,0)
##        frameR[line*20,:]= (0,255,0)    
        
    # Show the Undistorted images
    #cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
    #cv2.imshow('Normal', np.hstack([frameL, frameR]))

    # Convert from color(BGR) to gray
    grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(grayR,grayL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    # Using the WLS filter
    filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

##    # Resize the image for faster executions
##    dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

    # Filtering the Results with a closing filter
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

    # Colors map
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

    # Show the result for the Depth_image
    #cv2.imshow('Disparity', disp)
    #cv2.imshow('Closing',closing)
    #cv2.imshow('Color Depth',disp_Color)
    cv2.imwrite('Filtered Color Depth',filt_Color)

    # Mouse click
    cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
    
    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    
# Save excel
##wb.save("data4.xlsx")

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
