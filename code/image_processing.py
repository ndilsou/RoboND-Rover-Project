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


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160), invert=False, no_canvas=True):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
                   & (img[:, :, 1] > rgb_thresh[1]) \
                   & (img[:, :, 2] > rgb_thresh[2])

    if invert:
        above_thresh = ~above_thresh
    if no_canvas:
        is_not_canvas = (img[:, :, 0] != 0) \
                        & (img[:, :, 1] != 0) \
                        & (img[:, :, 2] != 0)
        above_thresh = is_not_canvas & above_thresh
    # Index the array of zeros with the boolean array and set to 1
    # we add a guard against modifying the black background.
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


def color_boundaries(img, hsv_lower, hsv_upper):
    # Create an array of zeros same xy size as img, but single channel

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    color_select = cv2.bitwise_and(img, img, mask=mask)
    return color_select


def rock_filter(img, hsv_lower=(80, 50, 50), hsv_upper=(100, 255, 255), ksize=(9, 9), sigmaX=10):
    """
    isolate rocks in the image by applying a gaussian kernel followed by a first two band thresholding in bgr.
    Finally the grayscalled image is thresholded on last time to binarize it.
    """
    blurred_rock = cv2.GaussianBlur(src=img, ksize=ksize, sigmaX=sigmaX)
    tresh_rock = color_boundaries(blurred_rock, hsv_lower, hsv_upper)
    gray_rock = cv2.cvtColor(tresh_rock, cv2.COLOR_RGB2GRAY)
    th, binary_rock = cv2.threshold(gray_rock, 0, 255, cv2.THRESH_BINARY)
    return binary_rock


def obstacle_filter(warped_img, navigable_img, ksize=(11,11)):
    kernel = np.ones(ksize, np.uint8)
    thresholded_nocanvas = color_thresh(warped_img, invert=True, no_canvas=True)
    gradient_nocanvas = cv2.morphologyEx(thresholded_nocanvas, cv2.MORPH_GRADIENT, kernel)
    thresholded_canvas = color_thresh(warped_img, invert=True, no_canvas=False)
    gradient_canvas = cv2.morphologyEx(thresholded_canvas, cv2.MORPH_GRADIENT, kernel)
    obstacles = gradient_nocanvas.copy()
    obstacles[(gradient_nocanvas != gradient_canvas) | (navigable_img == 1)] = 0

    return obstacles


def dominant_channel_filter(image, channel):
    channels = [0, 1, 2]
    channels.remove(channel)
    binary_img = ((image[:,:,channel] > image[:,:,channels[0]]) &
                  (image[:,:,channel] > image[:,:,channels[1]])).astype(np.int)
    return binary_img


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image

    return warped


def is_valid_angle(angle, etol):
    """return true if the angle is +- within etol."""
    if angle < etol or angle > (360 - etol):
        return True
    else:
        return False


def is_valid_image(roll, pitch, yaw, eroll, epitch):
    if is_valid_angle(roll, eroll) and is_valid_angle(pitch, epitch):
        return True
    else:
        return False


def delta_update(arr, x, y, z, delta, cap=255, floor=0):
    arr[x, y, z] = np.clip(arr[x, y, z] + delta, floor, cap)


def colorize_img(img, R, G, B):
    return np.dstack((img * R, img * G, img * B)).astype(np.float)


