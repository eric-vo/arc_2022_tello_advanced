from pathlib import Path

import numpy as np
import cv2 as cv
from djitellopy import Tello

arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)

# Read the IDs to pop from 'ids_to_pop.txt'
with open(Path('ids_to_pop.txt')) as f:
    ids_to_pop = f.read().splitlines()
print("IDs to pop:", ids_to_pop)

# The balloon the Tello is currently following
balloon_following = None

# If the Tello is spinning
spinning = False

# PID delta time
DELTA_TIME = 0.5

# Camera matrices
camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                          [0.000000, 919.018377, 351.238301],
                          [0.000000, 0.000000, 1.000000]])
distortion = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

# Initialize and connect to the Tello
tello = Tello()
tello.connect()

# Initialize the camera
tello.streamon()
frame_read = tello.get_frame_read()


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


fb_pid = PID(1, 0, 0)
lr_pid = PID(1, 0, 0)
yaw_pid = PID(1, 0, 0)
pop = 10
print("tello.takeoff()")

# Run this code while there are still balloons to pop
while ids_to_pop:
    # Find ArUco markers
    corners, ids, rejects = cv.aruco.detectMarkers(
        cv.cvtColor(frame_read.frame, cv.COLOR_BGR2GRAY),
        arucoDict
    )

    if balloon_following is None:
        # If the Tello is not following a balloon, check if it can see any
        # ArUco markers
        if ids:
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
        # Check if the IDs include the balloon the Tello is following
        if ids and balloon_following in ids:
            # Calculate the vectors to the balloon
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(
                corners[ids == balloon_following],
                0.1,
                camera_matrix,
                distortion
            )
            print(f"rvec: {rvec}, tvec: {tvec}")
            # tvec = (x, y, z)
            # move towards balloon
            fb_err = tvec[0] + pop  # ?
            fb_move = fb_pid.perform(fb_err)
            lr_err = tvec[1]  # ?
            lr_move = lr_pid.perform(lr_err)
            yaw_err = rvec  # ?
            yaw_move = yaw_pid.perform(yaw_err)
            print("tello.send_rc_control(lr_move, fb_move, 0, yaw_move)")



        else:
            # If the Tello is following a balloon but cannot see that
            # balloon, spin
            if not spinning:
                print("tello.send_rc_control(0, 0, 0, 50)")
                spinning = True
            ids_to_pop.remove(str(balloon_following))
            balloon_following = None
            fb_pid.reset()
            lr_pid.reset()
            yaw_pid.reset()

    # Display the frame
    cv.imshow('Tello Camera', frame_read.frame)

    # Quit if q is pressed
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
print("tello.land()")