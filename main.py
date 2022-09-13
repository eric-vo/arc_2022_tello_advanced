from pathlib import Path

import cv2 as cv
from djitellopy import Tello

hsv_ranges = {
    'RED': ((0, 100, 100), (10, 255, 255)),
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
ranges_to_detect = [hsv_ranges[color_name] for color_name in color_names]
print("Color names:", color_names)
print("Ranges to detect:", ranges_to_detect)

# # Initialize and connect to the Tello
# tello = Tello()
# tello.connect()

# # Initialize the camera
# tello.streamon()
# frame_read = tello.get_frame_read()

# while True:
#     # Get the frame
#     frame = frame_read.frame

#     # Convert the frame to HSV
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

#     # Detect the colors in the frame
#     for lower, upper in ranges_to_detect:
#         # Create a mask for the color
#         mask = cv.inRange(hsv, lower, upper)

#         # Find the contours
#         contours = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

#     # Display the frame
#     cv.imshow('Tello Camera', frame)

#     # Quit if q is pressed
#     if cv.waitKey(1) == ord('q'):
#         break

# cv.destroyAllWindows()