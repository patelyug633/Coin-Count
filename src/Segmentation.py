import cv2 
import numpy as np


import cv2
import numpy as np

# 1. Load image and convert to grayscale
img = cv2.imread('../dataset/images.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Blur to reduce noise and avoid false circle detections
gray_blurred = cv2.medianBlur(gray, 5)

# 3. Apply Hough Circle Transform
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1.2,             
    minDist=110,        # Prevents multiple detections on the same coin
    param1=100,         # Higher Canny edge threshold ignores faint inner details
    param2=55,          # Higher accumulator threshold requires a strong, complete circle
    minRadius=55,       # Filters out everything smaller than the smallest coin (quarter)
    maxRadius=95        # Caps the size just above the largest coin (half dollar)
)

# 4. Convert (x, y, r) to bounding boxes and draw them
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i, (x, y, r) in enumerate(circles[0, :]):
        # Bounding box coordinates: (top-left) and (bottom-right)
        x1, y1 = int(x - r), int(y - r)
        x2, y2 = int(x + r), int(y + r)

        # Crop the image: [y_start:y_end, x_start:x_end]
        cropped_img = img[y1:y2, x1:x2]

        # Save or display the extracted part
        cv2.imwrite(f"../dataset/RetrivedCoinImages/Coin{i+1}.jpg", cropped_img)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Detected Bounding Boxes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
