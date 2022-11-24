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

# Extra distance to travel into the balloon (meters)
POP_DISTANCE = 0.1

# Time before moving onto next marker or spinning again
WAITING_TIME = 1

# Degrees to spin (counter-clockwise) to search for balloons
SPIN_AMOUNT = 40
SPIN_LIMIT = 360
degrees_spun = 0

# PID delta time
DELTA_TIME = 0.2
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
fb_pid = PID(8, 2, 0)
lr_pid = PID(5, 20, 0)
ud_pid = PID(5, 15, 5)
yaw_pid = PID(5, 15, 2)

# PID errors
fb_err = 0
lr_err = 0
ud_err = 0
# yaw_err = 0

# Initialize and connect to the Tello
tello = Tello()
tello.connect()

tello.send_rc_control(0, 0, 0, 0)
print("Battery level: ", tello.get_battery())

# Initialize the camera
tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()

# Run this code while there are still balloons to pop
while True:
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

                    # Reset the time counter so there's no accidental landing
                    last_time = time.time()

                    degrees_spun = 0

                    break
            else:
                # Spin if no poppable markers are in sight
                if time.time() - last_time > WAITING_TIME:
                    tello.rotate_counter_clockwise(SPIN_AMOUNT)
                    last_time = time.time()
                    degrees_spun += SPIN_AMOUNT
        else:
            # If the Tello cannot see any ArUco markers, spin
            if time.time() - last_time > WAITING_TIME:
                tello.rotate_counter_clockwise(SPIN_AMOUNT)
                last_time = time.time()
                degrees_spun += SPIN_AMOUNT
    else:
        # Check if the IDs include the balloon the Tello is following
        if ids is not None and balloon_following in ids:
            # If it's time to perform PID
            if time.time() - last_time >= DELTA_TIME:
                # Get the corners of the balloon the Tello is following
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
                # print(f"rvec: {rvec}, tvec: {tvec}")
                # tvec = (x, y, z)

                # Move towards balloon
                fb_err = tvec[0][0][2] + POP_DISTANCE
                fb_move = fb_pid.perform(fb_err)

                lr_err = tvec[0][0][0]
                lr_move = lr_pid.perform(lr_err)

                ud_err = -tvec[0][0][1]
                ud_move = ud_pid.perform(ud_err)

                # Yaw error is inconsistent, so use the left-right error
                # yaw_err = -rvec[0][0][1]
                yaw_move = yaw_pid.perform(lr_err)

                tello.send_rc_control(round(lr_move), round(fb_move),
                                      round(ud_move), round(yaw_move))

                last_time = time.time()
        else:
            # If followed balloon is not seen for the waiting time
            if time.time() - last_time >= WAITING_TIME:
                # Reset movement and PID
                balloon_following = None

                fb_pid.reset()
                lr_pid.reset()
                ud_pid.reset()
                yaw_pid.reset()

                tello.send_rc_control(0, 0, 0, 0)

                last_time = time.time()

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

    # Put text indicating battery level
    cv.putText(
        frame_read.frame,
        f"Battery: {tello.get_battery()}",
        (10, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Put text with the errors on the right side of the screen
    cv.putText(
        frame_read.frame,
        f"FB: {round(fb_err, 2)}",
        (frame_read.frame.shape[1] - 200, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv.putText(
        frame_read.frame,
        f"LR: {round(lr_err, 2)}",
        (frame_read.frame.shape[1] - 200, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv.putText(
        frame_read.frame,
        f"UD: {round(ud_err, 2)}",
        (frame_read.frame.shape[1] - 200, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    # cv.putText(
    #     frame_read.frame,
    #     f"Yaw: {round(yaw_err, 2)}",
    #     (frame_read.frame.shape[1] - 200, 120),
    #     cv.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 0, 255),
    #     2
    # )

    # Display the frame
    cv.imshow("Tello Camera", frame_read.frame)

    # If the drone has spun at or past the spin limit, break
    if degrees_spun >= SPIN_LIMIT:
        break

    # Quit if q is pressed
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
tello.streamoff()
tello.land()
sys.exit()
