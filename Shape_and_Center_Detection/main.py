import cv2
import imutils

# read the image
image = cv2.imread("im_shape.png")
# resize the image
resized = imutils.resize(image, width=300)
# calculate ratio value
ratio = image.shape[0] / float(resized.shape[0])
# convert bgr to gray channel
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# apply Gaussian blurring
blurred = cv2.GaussianBlur(gray, (5,5), 0)
# apply threshold and set threshold value
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
# find contours
contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# make it suitable for all opencv versions
contour = imutils.grab_contours(contour)
for i in contour:
    # get moments of contours to find center
    M = cv2.moments(i)
    # find x and y values of the center
    ix = int((M["m10"] / M["m00"]) * ratio)
    iy = int((M["m01"] / M["m00"]) * ratio)
    # calculate the perimeter of the contour
    peri = cv2.arcLength(i, True)
    # get the number of vertices
    approx = cv2.approxPolyDP(i, 0.04 * peri, True)
    print(approx)
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        (x,y,w,h) = cv2.boundingRect(approx)
        oran = w / float(h)

        if oran == 1:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif len(approx) == 5:
        shape = "Pentagon"
    else:
        shape = "Circle"
    i = i.astype("float")
    i*=ratio
    i = i.astype("int")

    cv2.drawContours(image, [i], -1, (0,255,0), 2)
    # type shape
    cv2.putText(image, shape, (ix+5,iy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # draw around shapes
    cv2.circle(image, (ix, iy), 3, (0, 0, 0), -1)
cv2.imshow('Window',image)
cv2.waitKey(0)
