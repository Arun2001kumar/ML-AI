import numpy as np
import cv2 as cv
import glob

chessboardSize = (24,17)
frameSize = (1440,1080)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


objpoints = [] 
imgpoints = []

images = glob.glob('C:\images\calib\*.png')
for image in images:
    
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

   
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)


    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
 

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
img = cv.imread('C:\images\chess-demo.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)


x, y, w, h = roi
cv.imwrite('caliResult0.png', dst)
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibResult1.png', dst)


mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
