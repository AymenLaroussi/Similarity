import cv2

# Call images
img1 = cv2.imread('image/1.jpg', 0)
img2 = cv2.imread('image/2.jpg', 0)

orb = cv2.ORB_create()
# create a landmark
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)

# Display landmark images
cv2.imshow('kp1', imgKp1)
cv2.imshow('kp2', imgKp2)

# Display images
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
cv2.waitKey(0)


