"""
This module contains a set of function used essentially in the perception step.
Those functions focus on providing information that will help the Rover better navigate it's
environment.
"""

import numpy as np

VISITED_WEIGHT = 0.05 # Weight given to pixels already visited by the Rover.

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
    """
    Converts (dist, angle) from polar coordinate to cartesian coordinate.
    """
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
    """
    Mimics the concepts of a Point Cloud in 2d.
    Provide a mechanism to sense obstacle and their distance/angle.
    Performs a beam reading based on the current knowledge in the map.
    Each pixel surrounding the Rover's position in the binqry worldmap is captured.
    """
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
    """
    Returns the normal vector to two point.
    """
    dv = x - y
    normal = np.array([-dv[1], dv[0]])
    normal = normal / np.linalg.norm(normal, 2)
    return normal


def visit_location(Rover):
    """
    Flag the surrounding pixels as visited to make them less attractive in the field of vision.
    """
    trace = 1
    x, y = Rover.pos
    x = to_int(x)
    y = to_int(y)
    Rover.visited_map[y - trace:y + trace, x - trace: x + trace] = True


def weight_visited(Rover, nav_x_world, nav_y_world):
    """
    Check the current field of vision and weight each pixel based on whether it's
    been visited or not. This allow the Rover to focus attention around areas with
    new pixel.
    """
    if len(nav_x_world) > 0:
        weights = np.ones_like(nav_x_world, dtype=np.float)
        for i, (x, y) in enumerate(zip(nav_x_world, nav_y_world)):
            if Rover.visited_map[y, x]:
                weights[i] = VISITED_WEIGHT
        sum_w = weights.sum()
        if not np.isnan(sum_w).all():
            try:
                weights = weights / sum_w
            except RuntimeWarning as e:
                weights = np.ones_like(nav_x_world)

    else:
        weights = np.array([])
    Rover.nav_weights = weights
