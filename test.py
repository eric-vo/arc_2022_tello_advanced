import sys
import time
from pathlib import Path

import numpy as np
import cv2 as cv
from djitellopy import Tello

aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)

# Read the IDs to pop from 'ids_to_pop.txt'
with open(Path('ids_to_pop.txt')) as f:
    ids_to_pop = f.read().splitlines()
print("IDs to pop:", ids_to_pop)

# Camera matrices
CAMERA_MATRIX = np.array([[921.170702, 0.000000, 459.904354],
                          [0.000000, 919.018377, 351.238301],
                          [0.000000, 0.000000, 1.000000]])
DISTORTION = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

# The balloon ID the Tello is currently following
balloon_following = None

# If the Tello is spinning
spinning = False

# Extra distance to travel into the balloon
POP_DISTANCE = 10

# PID delta time
DELTA_TIME = 0.5
last_time = time.time()


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

    def reset(self):
        self.i = 0
        self.last_error = 0


# PID controllers
fb_pid = PID(1, 0, 0)
lr_pid = PID(1, 0, 0)
yaw_pid = PID(1, 0, 0)

# Initialize and connect to the Tello
tello = Tello()
tello.connect()

print("Battery level: ", tello.get_battery())

# Initialize the camera
tello.streamon()
frame_read = tello.get_frame_read()

print("tello.takeoff()")

# Run this code while there are still balloons to pop
while ids_to_pop:
    # Find ArUco markers
    corners, ids, rejects = cv.aruco.detectMarkers(
        cv.cvtColor(frame_read.frame, cv.COLOR_BGR2GRAY),
        aruco_dict
    )

    if balloon_following is None:
        # If the Tello is not following a balloon, check if it can see any
        # ArUco markers
        if ids is not None:
            # If the Tello can see any ArUco markers, check if any of them
            # are in the list of ids to pop
            for id in ids:
                if str(id[0]) in ids_to_pop:
                    # If the Tello can see a marker that it needs to pop,
                    # set the balloon it is following to that marker
                    balloon_following = id[0]

                    # Stop spinning if spinning
                    if spinning:
                        print("tello.send_rc_control(0, 0, 0, 0)")
                        spinning = False

                    break
                else:
                    # Spin if no poppable markers are in sight
                    if not spinning:
                        print("tello.send_rc_control(0, 0, 0, 50)")
                        spinning = True
        else:
            # If the Tello cannot see any ArUco markers, spin
            if not spinning:
                print("tello.send_rc_control(0, 0, 0, 50)")
                spinning = True
    else:
        # If the Tello is following a balloon, check if it can see any
        # ArUco markers
        if ids is not None:
            # If the Tello can see any ArUco markers, check if any of them
            # are the balloon it is following
            for i, id in enumerate(ids):
                if id[0] == balloon_following:
                    # If the Tello can see the balloon it is following,
                    # calculate the center of the marker
                    center = np.mean(corners[i][0], axis=0)

                    # Calculate the error in the x and y directions
                    x_error = center[0] - 480
                    y_error = center[1] - 360

                    # Calculate the yaw error
                    yaw_error = np.arctan2(x_error, y_error)

                    # Calculate the distance to the balloon
                    distance = np.linalg.norm(center - np.array([480, 360]))

                    # Calculate the PID values
                    fb = fb_pid.perform(distance)
                    lr = lr_pid.perform(x_error)
                    yaw = yaw_pid.perform(yaw_error)

        # Check if the IDs include the balloon the Tello is following
        if ids is not None and balloon_following in ids:
            # If enough time has passed
            if time.time() - last_time >= DELTA_TIME:
                corners_following = np.array(corners[
                    np.where(ids == balloon_following)[0][0]
                ])

                # Calculate the vectors to the balloon
                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(
                    corners_following,
                    0.1,
                    CAMERA_MATRIX,
                    DISTORTION
                )
                print(f"rvec: {rvec}, tvec: {tvec}")
                # tvec = (x, y, z)

                # Move towards balloon
                fb_err = tvec[0][0][2] + POP_DISTANCE
                fb_move = fb_pid.perform(fb_err)

                lr_err = tvec[0][0][0]
                lr_move = lr_pid.perform(lr_err)

                yaw_err = rvec[0][0][1]
                yaw_move = yaw_pid.perform(yaw_err)

                print("tello.send_rc_control(lr_move, fb_move, 0, yaw_move)")

                last_time = time.time()
        else:
            # If followed balloon is not seen for 1 second
            if time.time() - last_time >= DELTA_TIME * 2:
                # If the Tello is following a balloon but cannot see that
                # balloon, remove that balloon from the list and start spinning
                ids_to_pop.remove(str(balloon_following))
                balloon_following = None

                fb_pid.reset()
                lr_pid.reset()
                yaw_pid.reset()

                if not spinning:
                    print("tello.send_rc_control(0, 0, 0, 50)")
                    spinning = True

    # Outline detected markers
    cv.aruco.drawDetectedMarkers(
        frame_read.frame,
        corners,
        ids
    )

    # Put text indicating which balloon the Tello is following
    cv.putText(
        frame_read.frame,
        f"Following balloon: {balloon_following}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display the frame
    cv.imshow('Tello Camera', frame_read.frame)

    # Quit if q is pressed
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
tello.streamoff()
print("tello.land()")
sys.exit()
