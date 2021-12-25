import cv2

# Call images
img1 = cv2.imread('image/1.jpg', 0)
img2 = cv2.imread('image/2.jpg', 0)

orb = cv2.ORB_create()
# Display images
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)


