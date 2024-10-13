import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
#import findpeaks

# Aqcuire raw image
#im = PiCam.TakeSingleImage(showImage=False,saveImage=True)
im = cv2.imread('test_img_30.jpeg')
#im = cv2.imread('C:/Users/crk5344/Desktop/White_light-1.jpg')

#"E:\2024_07_15-17_36_image1.jpeg"
plt.imshow(im)
plt.title('Raw')
plt.show()
plt.close()

# Crop image
# Vertical
x1 = 250
x2 = 1750
# horizontal
y1 = 500
y2 = 2000
hInterval = y2-y1
vInterval = x2-x1
im = im[x1:x2,y1:y2]

im_raw = im

# Convert image
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#imgray = cv2.equalizeHist(imgray)
plt.imshow(imgray,cmap='gray',vmin=0,vmax=255)
plt.title('grayscale')
plt.show()
plt.close()


ret, binary = cv2.threshold(imgray,75,255,cv2.THRESH_BINARY)
#binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
#binary = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,251,15)
binary = 255-binary # avoid image frame as contour
# plt.imshow(binary,cmap='gray',vmin=0,vmax=255)
# plt.title('Binary')
# plt.show()
# plt.close()

# Find contours in binary image and display them
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im, contours, -1, (0,255,255), 3)
plt.figure()
plt.imshow(im, cmap='gray')
plt.title('Raw with contours')
plt.show()
plt.clf()
plt.close()

contour = np.array([list(pt[0]) for ctr in contours for pt in ctr])

cnts = []
for cnt in contours:
    cnt = np.squeeze(np.array(cnt, dtype=int), axis=1)
    cnts.append(cnt)

## Array of Shape N x 2, N: number of contours, 2: x,y
contourPoints = np.vstack((cnts))



# Delete points not interested in
# Delete in y (vertical)
idx = np.where(contourPoints[:,1]<10)
contourPoints = np.delete(contourPoints,idx,0)
idx = np.where(contourPoints[:,1]> vInterval-10)
contourPoints = np.delete(contourPoints,idx,0)
#
# # Delete in x (horizontal)
idx = np.where(contourPoints[:,0] <10)
contourPoints = np.delete(contourPoints,idx,0)
idx = np.where(contourPoints[:,0] >hInterval-10)
contourPoints = np.delete(contourPoints,idx,0)


pointsX = contourPoints[:,0]
pointsY = contourPoints[:,1]

# sort points
idx   = np.argsort(pointsX)
pointsX = np.array(pointsX)[idx]
pointsY = np.array(pointsY)[idx]



# Analysze points to detect flap wheel start and end
countsX, binsX = np.histogram(pointsX, bins=250)
binX_centers = (binsX[:-1] + binsX[1:]) / 2

countsX = np.concatenate([countsX,np.zeros((100))])
peaks,_ = scipy.signal.find_peaks(countsX,height=75,distance=50)
peak_heights = countsX[peaks]

sorted_peaks = np.argsort(peak_heights)[::-1]  # sortiere absteigend
top_two_peaks = peaks[sorted_peaks[:2]]


# Endpunkte des Histogramms berücksichtigen
potential_peaks = np.concatenate(([0], peaks, [len(countsX)-1]))
potential_heights = countsX[potential_peaks]
sorted_potential_peaks = np.argsort(potential_heights)[::-1]
top_two_peaks_with_ends = potential_peaks[sorted_potential_peaks[:2]]

countsX = countsX[0:250]

plt.hist(countsX,binsX,color='g')
plt.plot(binX_centers,countsX)
plt.scatter(binX_centers[top_two_peaks], countsX[top_two_peaks], s=1)
plt.axvline(binsX[top_two_peaks[0]],color='r')
plt.axvline(binsX[top_two_peaks[1]],color='r')
plt.xlabel('Datenwerte')
plt.ylabel('Häufigkeit')
plt.grid()
plt.show()

xleft = min(top_two_peaks)
xright = max(top_two_peaks)

#im_raw[contourPoints[:,1],contourPoints[:,0]] = [255,0,0] # Draw contour
im_raw[:, np.round(binX_centers[xleft].astype(int))] =  [255,0,0] # Draw vertical line for flap end (right)
im_raw[:, np.round(binX_centers[xright].astype(int))] =  [255,0,255] # Draw vertical line for flap end (lef)


# Delete contourpoints left and right outside of flap wheel
idx = np.where(pointsX < binX_centers[xleft])
pointsX = np.delete(pointsX,idx,0)
pointsY = np.delete(pointsY,idx,0)
idx = np.where(pointsX > binX_centers[xright])
pointsX = np.delete(pointsX,idx,0)
pointsY = np.delete(pointsY,idx,0)



y_rotaxis = np.mean(pointsY)

plt.figure()
plt.imshow(im_raw)
plt.scatter(pointsX,pointsY,color='r',s=1) # raw profile
plt.axvline(binX_centers[xleft],color='b',linestyle='-',linewidth=1) # vertical lines
plt.axvline(binX_centers[xright],color='b',linestyle='-',linewidth=1) # vertical lines
plt.axhline(y_rotaxis,color='b',linestyle='--',linewidth=2) #rotation axis
plt.title('Filtered contours')
plt.show()
plt.close()




# Calculate discrete intermediate points
x_profile = np.linspace(binX_centers[xleft+5],binX_centers[xright-5], num=10, endpoint=True).astype(int)
y_profile = np.zeros([len(x_profile),1],dtype=int)


for i in range(0,len(x_profile)):
    idx = np.where(x_profile[i]==pointsX)
    if len(pointsY[idx]) >1:
        y_profile[i] = max(pointsY[idx])-y_rotaxis
    else:
        y_profile = pointsY[idx] -y_rotaxis


x_profile = x_profile-x_profile[0]


plt.plot(x_profile,y_profile) # plot interval points
plt.grid()
plt.xlim((0,max(x_profile)+10))
plt.ylim((0,max(y_profile)+10))
plt.axis('equal')
plt.xlabel('axial')
plt.ylabel('radial')
plt.show()