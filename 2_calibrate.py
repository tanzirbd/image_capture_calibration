import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (4,5) #define the number of chessboard pattern (row,column)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
correct_idx = [] #correct_image_save

images = glob.glob('8_chessboard_two_camera/chessboard_two/right/*.png')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    p=0
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (4,5), corners2,ret)
#        cv2.putText(img, str(idx), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        p=str(p)
        #cv2.imwrite('report/capture_640x480/left/'+p+'.png',img)
      
#        key = cv2.waitKey(1000)
#        if key & 0xFF == ord('a'):
#            correct_idx.append(idx)
#            objpoints.append(objp)
#            imgpoints.append(corners2)
#            continue
#        else:
#            continue

#print(correct_idx)
    p=int(p)+1
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, K, D, rvecs, tvecs = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print('\nSide-View')
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
#print(str(rvecs))
#print(str(tvecs))
np.savez('9_calibration_parameter/right.npz',DIM=_img_shape[::-1],K=K,D=D,rvecs=rvecs,tvecs=tvecs,imgpoints=imgpoints)

