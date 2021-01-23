import cv2
import numpy as np
import os
import glob
import sys

#left_cam
DIM=(1600, 1200)
#K=np.array([[592.647370581156, 0.0, 789.241817495145], [0.0, 592.4843549371146, 554.834903597127], [0.0, 0.0, 1.0]])
#D=np.array([[-0.07242285117478836], [0.0822765377853764], [-0.09526587504179218], [0.03472086627984029]])
#right
K=np.array([[572.8418168729697, 0.0, 768.8933846554007], [0.0, 566.4671123237823, 569.8724445703905], [0.0, 0.0, 1.0]])
D=np.array([[-0.16794173956920846], [0.32649496603067707], [-0.32457028678195465], [0.1061819429850892]])

#left
#K=np.array([[574.252713803618, 0.0, 821.1686800742988], [0.0, 578.3712283518574, 560.4496954438866], [0.0, 0.0, 1.0]])
#D=np.array([[-0.24658282153180597], [0.49829650345732374], [-0.5015356440477023], [0.1666060923013115]])
#Side-View
#Found 120 valid images for calibration
DIM=(1600, 1200)
#K=np.array([[560.3496469844627, 0.0, 797.4401137301651], [0.0, 565.2708604172439, 620.635825547462], [0.0, 0.0, 1.0]])
#D=np.array([[-0.05812798882134729], [0.03210453823497408], [-0.03274943813682195], [0.01050921965983058]])

#right
#K=np.array([[571.5341612073496, 0.0, 800.3296263713172], [0.0, 571.1949827321278, 561.9696207900334], [0.0, 0.0, 1.0]])
#D=np.array([[-0.043296433215560676], [-0.012458201439613313], [0.006414600004321073], [-0.0016142863609245896]])




#K=np.array([[826.53409696,   0.0,         790.83335041],
# [  0.0,         808.2649773,  600.71940929],
# [  0.0,           0.0,           1.0        ]])
#D=np.array([[-0.60930172],  [0.62141923], [-0.00579851],  [0.01397509], [-0.31105334]])

#files=np.load('WorkPlace/20_05_08_newdata/chessboard/left_calib.npz')
#K=files['K']
#D=files['D']
print(K,'\n')
#print(D)


#right_cam
#DIM=(1600, 1200)
#K=np.array([[566.7145103702243, 0.0, 798.446140753528], [0.0, 564.8935246208073, 562.2753444249794], [0.0, 0.0, 1.0]])
#D=np.array([[-0.04531604494188637], [-0.01309158350731869], [0.008543281251721446], [-0.0023284737282867327]])


def undistort(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    print(dim1)
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    frame1 = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    print(new_K)
    cv2.imwrite("frame_undis_right.png", frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
