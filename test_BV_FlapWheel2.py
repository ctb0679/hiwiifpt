import cv2
import numpy as np
from matplotlib import pyplot as plt


# Read image
im = cv2.imread('/home/junaidali/catkin_ws/src/hiwiifpt/image129.jpeg')

# Crop image
x1 = 250
x2 = 1000
y1 = 250
y2 = 1750
im = im[x1:x2,y1:y2]

# Detect contours
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(imgray,127,255,0)
binary = 255-binary # avoid image frame as contour

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(im, contours[0], -1, (0,255,0), 3)
#cv2.drawContours(im, contours[1], -1, (255,0,0), 3)
plt.imshow(binary)
plt.show()


#print(contours[0][2,0,1])

#print(contours[0].shape[0])

points = np.zeros((contours[0].shape[0],2))

# Array with Contours
for j in range(contours[0].shape[0]):
    points[j][0] = contours[0][j,0,0]
    points[j][1] = contours[0][j,0,1]

points = (np.rint(points)).astype(int)

# Delete points not interested in
idx = np.where(points == 0)
points = np.delete(points,idx,0)


idx = np.where(points[:,1]>700)
points = np.delete(points,idx,0)

idx = np.where(points[:,0] <400)
points = np.delete(points,idx,0)

pointsX = points[:,0]
pointsY = points[:,1]

# sort points
idx   = np.argsort(pointsX)
pointsX = np.array(pointsX)[idx]
pointsY = np.array(pointsY)[idx]

im[points[:,1],points[:,0]] = [255,0,0]

plt.imshow(im)
plt.show()

x_start=0
x_end = y2-y1


pointsY = -pointsY+abs(min(-pointsY))

# Plot
plt.scatter(pointsX,pointsY) # raw profile

x_profile = np.linspace(min(pointsX),max(pointsX), num=20, endpoint=True).astype(int)
y_profile = np.zeros([len(x_profile),1],dtype=int)

#idx = np.where(pointsX == x_profile)

for i in range(0,len(x_profile)):
    for j in range(0,len(pointsX)):
        if x_profile[i] == pointsX[j]:
            y_profile[i] = pointsY[j]


plt.scatter(x_profile,y_profile) # plot interval points
plt.grid()
plt.xlim((400,1400))
plt.ylim((0,1000))
plt.xlabel('axial')
plt.ylabel('radial')
plt.show()
