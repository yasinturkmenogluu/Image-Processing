# import necessary libraries
import cv2
import numpy as np

# create cap object
cap = cv2.VideoCapture(0)

while True:
    # read camera
    ret, frame = cap.read()
    
    # blurred the image with gaussian and median blur
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    blurred_frame = cv2.medianBlur(blurred_frame, 5)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    
    # Set color
    colors = {"Orange": (255,255,255)}
    lower = {"Orange": (5,105,105)}
    upper = {"Orange": (15,255,255)}

    for key, value in upper.items():
        # create a kernel
        kernel = np.ones((7, 7), np.uint8)
        
        # Apply Threshold with inRange
        PrimaryMask = cv2.inRange(hsv, lower[key], upper[key])
        
        # Apply morphologyEx mask
        mask = cv2.morphologyEx(PrimaryMask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(PrimaryMask, cv2.MORPH_CLOSE, kernel)
        
        center = None
        # find contours
        _ ,contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            # Set central points
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            
            if len(contours) > 0:
                area = cv2.contourArea(contour)
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if (area > 300):  # for removing small noises
                    M = cv2.moments(c)
                    # find center point
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.drawContours(frame, [approx], -1, colors[key], 2)
                    cv2.circle(frame, center, 3, (0, 0, 0), 2)

                    print("center",center)
                    cv2.circle(frame, center, 7, (0, 0, 255), -1)
                    cv2.putText(frame, key, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[key])

   # cv2.imshow('Masked image', mask)
    cv2.imshow('frame', frame)

    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
