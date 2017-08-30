import numpy as np
import cv2

def calibrate_image(image):
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - 2 * dst_size - bottom_offset],
                              ])
    return source, destination


def color_boundaries(img, hsv_lower, hsv_upper):
    # Create an array of zeros same xy size as img, but single channel

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    color_select = cv2.bitwise_and(img, img, mask=mask)
    return color_select


def rock_filter(img, hsv_lower=(80, 50, 50), hsv_upper=(100, 255, 255), ksize=(5, 5), sigmaX=5):
    """
    isolate rocks in the image by applying a gaussian kernal followed by a first two band thresholding in bgr.
    Finaly the grayscalled image is thresholded on last time to binarize it.
    """
    blurred_rock = cv2.GaussianBlur(src=img, ksize=ksize, sigmaX=sigmaX)
    tresh_rock = color_boundaries(blurred_rock, hsv_lower, hsv_upper)
    gray_rock = cv2.cvtColor(tresh_rock, cv2.COLOR_RGB2GRAY)
    th, binary_rock = cv2.threshold(gray_rock, 0, 255, cv2.THRESH_BINARY)
    return binary_rock

def to_cartesian_coords(dist, angle):
    x = dist * np.cos(angle)
    y = dist * np.sin(angle)
    return x, y


def is_valid_angle(angle, etol):
    """return true if the angle is +- within etol."""
    if angle < etol or angle > (360 - etol):
        return True
    else:
        return False


def is_valide_image(roll, pitch, yaw, eroll, epitch, eyaw):
    if is_valid_angle(roll, eroll) and is_valid_angle(pitch, epitch) and is_valid_angle(yaw, eyaw):
        return True
    else:
        return False


def delta_update(arr, x, y, z, delta, cap=255, floor=0):
    arr[x, y, z] = np.clip(arr[x, y, z] + delta, floor, cap)


def colorize_img(img, R, G, B):
    return np.dstack((img * R, img * G, img * B)).astype(np.float)


def find_rock(binary_rock, rock_radius, src, dst):
    """
    returns the location of the rock as a circle with radius as defined in arguments.
    """
    # initialised empty
    xrock = np.array([])
    yrock = np.array([])

    warped_rock = perspect_transform(binary_rock, src, dst)
    y, x = warped_rock.nonzero()
    if y.any() and x.any():
        rock_idx = np.argmax(y)
        xrock, yrock = x[rock_idx], y[rock_idx]
        rock_circle = np.zeros_like(warped_rock)
        cv2.circle(rock_circle, (np.uint8(xrock), np.uint8(yrock)), rock_radius, (255, 255, 255), -1)
        xrock, yrock = rover_coords(rock_circle)

    return xrock, yrock