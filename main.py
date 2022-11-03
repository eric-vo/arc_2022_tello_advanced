from pathlib import Path

import numpy as np
import cv2 as cv
from djitellopy import Tello

arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)

DELTA_TIME = 0.1

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

colors_and_ids = {}

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
marker_found = False


class PID:
    def __init__(self, kP, kI, kD):
        self.kP = kP
        self.kI = kI
        self.kD = kD

        self.i = 0
        self.last_error = 0

    def perform(self, error):
        self.p = self.kP * error
        self.i += self.kI * error * DELTA_TIME
        self.d = self.kD * (error - self.last_error) / DELTA_TIME
        self.last_error = error
        return self.p + self.i + self.d


i = 0
# Run this code while there are still balloons to pop
while color_names:
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

    # If a contour was not found
    if largest_contour is None:
        if color_found:
            print("Switching to next color (assuming balloon popped)")
            i += 1
            color_found = False
            continue
        else:
            # Make the drone spin if no color found yet
            # tello.send_rc_control(0, 0, 0, 10)
            pass
    else:
        # Draw the contour
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

    # Find ArUco markers
    corners, ids, rejects = cv.aruco.detectMarkers(
        cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
        arucoDict
    )

    if not marker_found:
        # Find the centroid of the markers if markers are found
        centroids = []
        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]
                centroids.append(np.mean(c, axis=0))

        # Find the marker centroid closest to the contour
        if centroids and largest_contour is not None:
            contour_centroid = [center_x, center_y]
            centroid_distances = [
                # Calculates distance between two points
                np.linalg.norm(marker_centroid - contour_centroid)
                for marker_centroid in centroids
            ]
            closest_marker_index = np.argmin(centroid_distances)
            marker_found = True

            # Associate the color with the marker ID
            colors_and_ids[color_names[i]] = ids[closest_marker_index]

    # Display the frame
    cv.imshow('Tello Camera', frame)

    # Quit if q is pressed
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()

# Create a file with color names and marker IDs
# In the format "COLOR-TAG_NUMBER"
# With filename 'colors_and_ids.txt'
with open(Path('colors_and_ids.txt'), 'w') as f:
    for color_name, marker_id in colors_and_ids.items():
        f.write(f"{color_name}-{marker_id[0]}")
