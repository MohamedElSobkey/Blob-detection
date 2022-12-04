import matplotlib.pyplot as plt 
import cv2


img = cv2.imread('blobs.jpg', 0)

#set up the SimpleBlobdetector with default parameters
params = cv2.SimpleBlobDetector_Params()


# Define thresholds
params.minThreshold = 0
params.maxThreshold = 255


# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.maxArea = 10000

# Filter by Color
params.filterByColor = False
params.blobColor = 0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1


# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1
params.maxConvexity = 1

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

print(' No. of blobs detected are : ', len(keypoints))
# Draw blobs
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None , (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_with_keypoints)
cv2.imshow('keypoints', img_with_keypoints )

cv2.waitKey(0)
cv2.destroyAllWindows()

