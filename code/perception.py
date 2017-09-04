import numpy as np
import cv2
import image_processing as img_helper
import navigation as nav_helper
from config import Constants

# Define a function to convert from image coords to rover coords
from networkx.generators.classic import full_rary_tree


def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel


def find_rock(warped_rock, rock_radius=Constants.ROCK_RADIUS):
    """
    returns the location of the rock as a circle with radius as defined in arguments.
    """
    # initialised empty
    xrock = np.array([])
    yrock = np.array([])
    xrock_center = np.array([])
    yrock_center = np.array([])
    y, x = warped_rock.nonzero()
    if y.any() and x.any():
        rock_idx = np.argmax(y)
        xrock_center, yrock_center = x[rock_idx], y[rock_idx]
        rock_circle = np.zeros_like(warped_rock)
        cv2.circle(rock_circle, (np.uint8(xrock_center), np.uint8(yrock_center)),
                   rock_radius, (255, 255, 255), -1)
        xrock, yrock = rover_coords(rock_circle)

    return xrock, yrock, xrock_center, yrock_center


def update_beams_reading(Rover):
    xpos, ypos = Rover.pos
    yaw = Rover.yaw
    obstacle_channel = img_helper.dominant_channel_filter(Rover.worldmap, 0)
    beam_points = nav_helper.find_beam_points(obstacle_channel, Rover.beam_angles, xpos, ypos, yaw,
                                              Constants.DELTA, atol=1e-1)
    Rover.beam_points = {angle: beam for angle, beam in zip(Rover.beam_angles, beam_points)}

    if Rover.beam_points[90]:
        beam_point = np.array(Rover.beam_points[90])
        normal_vector = -nav_helper.get_normal_vector(beam_point, 0)
        _, wall_angle = nav_helper.to_polar_coords(*normal_vector)
        Rover.wall_angle = wall_angle
    else:
        Rover.wall_angle = np.nan

    closest_point = (np.inf, None)
    furthest_point = (-np.inf, None)
    for beam_point in beam_points:
        if beam_point:
            dist, angle = nav_helper.to_polar_coords(*beam_point)
            if dist < closest_point[0]:
                # We want to point away from the obstacle
                closest_point = (dist,  angle)
            if dist > furthest_point[0]:
                furthest_point = (dist, angle)
    Rover.closest_obstacle = closest_point
    Rover.furthest_obstacle = furthest_point

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    img = Rover.img
    xpos, ypos = Rover.pos
    roll = Rover.roll
    pitch = Rover.pitch
    yaw = Rover.yaw

    img_size = img.shape[0]
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    src, dst = img_helper.calibrate_image(Rover.img)

    # 2) Apply perspective transform
    warped_image = img_helper.perspect_transform(img, src, dst)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    binary_rock = img_helper.rock_filter(img)
    binary_nav = img_helper.color_thresh(warped_image)
    binary_obstacle = img_helper.color_thresh(warped_image, invert=True)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = binary_obstacle*255
    Rover.vision_image[:, :, 1] = binary_rock*255
    Rover.vision_image[:, :, 2] = binary_nav*255
    # 5) Convert map image pixel values to rover-centric coords.replace(".",",")
    x_nav, y_nav = rover_coords(binary_nav)
    x_obstacle, y_obstacle = rover_coords(binary_obstacle)
    warped_rock = img_helper.perspect_transform(binary_rock, src, dst)
    x_rock, y_rock, x_rock_center, y_rock_center = find_rock(warped_rock, Constants.ROCK_RADIUS)
    if x_rock_center.any() and y_rock_center.any():
        Rover.seeing_sample = True
    else:
        Rover.seeing_sample = False

    Rover.sample_pos = x_rock_center, y_rock_center
    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_x_world, obstacle_y_world = nav_helper.pix_to_world(x_obstacle, y_obstacle,
                                                                 xpos, ypos, yaw, img_size, Constants.SCALE)
    nav_x_world, nav_y_world = nav_helper.pix_to_world(x_nav, y_nav,
                                                       xpos, ypos, yaw, img_size, Constants.SCALE)
    rock_x_world, rock_y_world = nav_helper.pix_to_world(x_rock, y_rock,
                                                         xpos, ypos, yaw, img_size, Constants.SCALE)


    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # update obstacle map
    if img_helper.is_valid_image(roll, pitch , yaw,
                                 eroll=Constants.ERR_ROLL, epitch=Constants.ERR_PITCH, eyaw=Constants.ERR_YAW):

        img_helper.delta_update(Rover.worldmap, obstacle_y_world, obstacle_x_world, 0, Constants.DELTA)

        # update navigation
        img_helper.delta_update(Rover.worldmap, nav_y_world, nav_x_world, 2, Constants.DELTA)
        # reduce certainty about obstacle.
        img_helper.delta_update(Rover.worldmap, nav_y_world, nav_x_world, 0, -Constants.DELTA * .66)

        # update rock sample map
        img_helper.delta_update(Rover.worldmap, rock_y_world, rock_x_world, 1, Constants.DELTA)
        # Rock area should be assumed navigable
        img_helper.delta_update(Rover.worldmap, rock_y_world, rock_x_world, 0, -Constants.DELTA)


    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    dists, angles = nav_helper.to_polar_coords(x_nav, y_nav)
    Rover.nav_dists = dists
    Rover.nav_angles = angles

    update_beams_reading(Rover)
    return Rover