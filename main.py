from pathlib import Path

import cv2 as cv
from djitellopy import Tello

MIN_CONTOUR_AREA = 50

HSV_RANGES = {
    'RED': ((172, 200, 150), (178, 255, 255)),
    'ORANGE': ((11, 100, 100), (25, 255, 255)),
    'YELLOW': ((26, 100, 100), (34, 255, 255)),
    'LIGHT_GREEN': ((35, 100, 100), (77, 255, 255)),
    'GREEN': ((35, 100, 100), (77, 255, 255)),
    'LIGHT_BLUE': ((78, 100, 100), (99, 255, 255)),
    'BLUE': ((100, 100, 100), (124, 255, 255)),
    'PURPLE': ((125, 100, 100), (155, 255, 255)),
    'PINK': ((156, 100, 100), (179, 255, 255)),
}

# Read the colors to pop from colors.txt
with open(Path('colors.txt')) as f:
    color_names = f.read().splitlines()
ranges_to_detect = [HSV_RANGES[color_name] for color_name in color_names]
print("Color names:", color_names)
print("Ranges to detect:", ranges_to_detect)

# # Initialize and connect to the Tello
# tello = Tello()
# tello.connect()

# # Initialize the camera
# tello.streamon()
# frame_read = tello.get_frame_read()

cap = cv.VideoCapture(0)
color_found = False
i = 0

while i < len(ranges_to_detect):
    # Get the current color to track
    current_lower, current_upper = ranges_to_detect[i]

    # Get the frame
    ret, frame = cap.read()
    # Convert the frame to HSV
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Create a mask for the color
    mask = cv.inRange(frame_hsv, current_lower, current_upper)

    # Find the contours
    contours = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

    # Find the largest contour
    largest_contour = None
    if contours:
        contour_areas = [cv.contourArea(contour) for contour in contours]
        greatest_contour_area = max(contour_areas)

        if greatest_contour_area >= MIN_CONTOUR_AREA:
            for contour in contours:
                if cv.contourArea(contour) == greatest_contour_area:
                    largest_contour = contour
                    break

    if largest_contour is None:
        if color_found:
            print("Color not found")
            i += 1
            color_found = False
            continue
    else:
        cv.drawContours(frame, [largest_contour], 0, (0, 255, 0), 3)
        color_found = True

    # Find the center of the contour
    if largest_contour is not None:
        moments = cv.moments(largest_contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            cv.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)

    frame_masked = cv.bitwise_and(frame, frame, mask=mask)
    
    # Display the frame
    cv.imshow('Tello Camera', frame)

    # Quit if q is pressed
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()