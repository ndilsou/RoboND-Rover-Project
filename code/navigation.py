import numpy as np
import cv2

import image_processing as img_helpers


def deg_to_rad(degree):
    return degree * np.pi / 180


def rad_to_deg(radian):
    return radian * 180 / np.pi


def norm(x, y):
    return np.sqrt(x ** 2 + y ** 2)

def to_int(f):
    return np.int(np.round(f))

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


def to_cartesian_coords(dist, angle):
    x = dist * np.cos(angle)
    y = dist * np.sin(angle)
    return x, y


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


def world_to_pix(x_pix_world, y_pix_world, xpos, ypos, yaw, scale):
    """
    Reverse the pixel transform, converting from world frame to robot frame
    """
    # Apply translation
    xworld_tran, yworld_tran = translate_pix(x_pix_world, y_pix_world, -ypos * scale, -xpos * scale, 1 / scale)
    # Apply rotation
    xworld_rot, yworld_rot = rotate_pix(xworld_tran, yworld_tran, -yaw)

    # Perform rotation, translation and clipping all at once
    xpix = xworld_rot
    ypix = yworld_rot
    # Return the result
    return xpix, ypix


def find_beam_points(worldmap, degrees, xpos, ypos, yaw, scale, atol):
    x_pix_world, y_pix_world = worldmap.nonzero()
    x_robot, y_robot = world_to_pix(x_pix_world, y_pix_world, xpos, ypos, yaw, scale)
    map_dists, map_angles = to_polar_coords(x_robot, y_robot)

    beam_points = np.empty((len(degrees), 2))
    for i, degree in enumerate(degrees):
        beam_filter = np.isclose(map_angles, deg_to_rad(degree), atol=atol)
        try:
            beam_index = np.argmin(map_dists[beam_filter])
            beam_polar_coord = [map_dists[beam_filter][beam_index], map_angles[beam_filter][beam_index]]
            beam_point = to_cartesian_coords(*beam_polar_coord)
        except ValueError:
            beam_point = [np.nan, np.nan]
        beam_points[i, :] = beam_point
    return beam_points


def get_normal_vector(x, y):
    dv = x - y
    normal = np.array([-dv[1], dv[0]])
    normal = normal / np.linalg.norm(normal, 2)
    return normal


def visit_location(Rover):
    trace = 1
    x, y = Rover.pos
    x = to_int(x)
    y = to_int(y)
    Rover.visited_map[y - trace:y + trace, x - trace: x + trace] = True


def weight_visited(Rover, nav_x_world, nav_y_world):
    weights = np.ones_like(nav_x_world)
    for i, (x, y) in enumerate(zip(nav_x_world, nav_y_world)):
        if Rover.visited_map[y, x]:
            weights[i] = 0.01
    Rover.nav_weights = weights / weights.sum()

