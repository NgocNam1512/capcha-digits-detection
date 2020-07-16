import cv2
import numpy as np
import os
import imutils
import pytesseract

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel5 = np.ones((5,5),np.uint8)
kernel7 = np.ones((7,7), np.uint8)
kernel9 = np.ones((9,9), np.uint8)

list_image = os.listdir('data')
for image in list_image:
    image = cv2.imread('data/'+image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 5)
    
    dialation = cv2.dilate(blur, kernel5, iterations=1)
    erosion = cv2.erode(dialation, kernel7, iterations=1)
    thresh = cv2.threshold(erosion, 150, 255, cv2.THRESH_BINARY_INV)[1] 

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts_sorted = sorted(cnts, key=lambda x: cv2.contourArea(x),reverse=True)
    top_cnts = []
    # remove too small region
    for cnt in cnts_sorted[:4]:
        if (cv2.contourArea(cnt) > 120):
            top_cnts.append(cnt)

    # get bounding box of top contours
    output = image.copy()
    # for (i, c) in enumerate(top_cnts):
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(output, (x,y), (x+w, y+h), (0,0,255), 2)

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for (i, c) in enumerate(top_cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        x1.append(x)
        x2.append(x+w)
        y1.append(y)
        y2.append(y+h)

    xmin = min(x1)
    ymin = min(y1)
    xmax = max(x2)
    ymax = max(y2)
    
    # _, rowbbox = sort_contours(top_cnts, "left-to-right")
    # xmin = rowbbox[0][0] # x
    # xmax = rowbbox[-1][0] + rowbbox[-1][2] # x+w

    # _, colbbow = sort_contours(top_cnts, "top-to-bottom")
    # ymin = colbbow[0][1]
    # ymax = colbbow[-1][0] + colbbow[-1][3] # y+h
    # cv2.drawContours(output, top_cnts, -1, (0,255,0), 2)
    cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0,0,255), 2)

    cv2.imshow("erosion", erosion)
    cv2.waitKey()
    cv2.imshow("image", output)
    cv2.waitKey()