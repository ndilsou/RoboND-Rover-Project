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
    y, x = warped_rock.nonzero()
    if y.any() and x.any():
        rock_idx = np.argmax(y)
        xrock_center, yrock_center = x[rock_idx], y[rock_idx]
        rock_circle = np.zeros_like(warped_rock)
        cv2.circle(rock_circle, (np.uint8(xrock_center), np.uint8(yrock_center)),
                   rock_radius, (255, 255, 255), -1)
        xrock, yrock = rover_coords(rock_circle)

    return xrock, yrock


def update_beams_reading(Rover):
    xpos, ypos = Rover.pos
    yaw = Rover.yaw
    obstacle_channel = img_helper.dominant_channel_filter(Rover.worldmap, 0)
    beam_points = nav_helper.find_beam_points(obstacle_channel, Rover.beam_angles, xpos, ypos, yaw,
                                              Constants.DELTA, atol=1e-1)

    if not np.isnan(beam_points).all():
        idx = np.argmin(np.linalg.norm(beam_points, axis=1))
        Rover.wall_point = beam_points[idx]
        if not np.isnan(Rover.wall_point).all():
            normal_vector = -nav_helper.get_normal_vector(Rover.wall_point, np.zeros(2))
            _, wall_angle = nav_helper.to_polar_coords(*normal_vector)
            Rover.wall_angle = wall_angle
        Rover.beam_points = {angle: beam for angle, beam in zip(Rover.beam_angles, beam_points)}
    else:
        Rover.wall_point = np.nan
        Rover.wall_angle = None


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    nav_helper.visit_location(Rover)
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
    # binary_obstacle = img_helper.obstacle_filter(warped_image, binary_nav)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = binary_obstacle*255
    Rover.vision_image[:, :, 2] = binary_nav*255
    # 5) Convert map image pixel values to rover-centric coords.replace(".",",")
    x_nav, y_nav = rover_coords(binary_nav)
    # kernel = np.ones((3, 3), np.uint8)
    # gradient = cv2.morphologyEx(binary_obstacle, cv2.MORPH_GRADIENT, kernel)
    x_obstacle, y_obstacle = rover_coords(binary_obstacle)
    warped_rock = img_helper.perspect_transform(binary_rock, src, dst)
    Rover.vision_image[:, :, 1] = warped_rock * 255

    x_rock, y_rock = find_rock(warped_rock, Constants.ROCK_RADIUS)

    if x_rock.any() and y_rock.any():
        Rover.seeing_sample = True
        r_dists, r_angles = nav_helper.to_polar_coords(x_rock, y_rock)
        Rover.curr_sample_pos = r_dists, r_angles
    else:
        Rover.seeing_sample = False



    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_x_world, obstacle_y_world = nav_helper.pix_to_world(x_obstacle, y_obstacle,
                                                                 xpos, ypos, yaw,
                                                                 Constants.WORLD_SIZE, Constants.SCALE)
    nav_x_world, nav_y_world = nav_helper.pix_to_world(x_nav, y_nav,
                                                       xpos, ypos, yaw,
                                                       Constants.WORLD_SIZE, Constants.SCALE)
    rock_x_world, rock_y_world = nav_helper.pix_to_world(x_rock, y_rock,
                                                         xpos, ypos, yaw,
                                                         Constants.WORLD_SIZE, Constants.SCALE)


    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # update obstacle map
    if img_helper.is_valid_image(roll, pitch, yaw,
                                 eroll=Constants.ERR_ROLL, epitch=Constants.ERR_PITCH):

        img_helper.delta_update(Rover.worldmap, obstacle_y_world, obstacle_x_world, 0, Constants.DELTA)
        img_helper.delta_update(Rover.worldmap, obstacle_y_world, obstacle_x_world, 2, -Constants.DELTA * 0.3)
        # update navigation
        img_helper.delta_update(Rover.worldmap, nav_y_world, nav_x_world, 2, 2 * Constants.DELTA)
        # reduce certainty about obstacle.
        img_helper.delta_update(Rover.worldmap, nav_y_world, nav_x_world, 0, -Constants.DELTA)

        # update rock sample map
        img_helper.delta_update(Rover.worldmap, rock_y_world, rock_x_world, 1, Constants.DELTA)


    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    nav_helper.weight_visited(Rover, nav_x_world, nav_y_world)
    dists, angles = nav_helper.to_polar_coords(x_nav, y_nav)
    Rover.nav_dists = dists
    Rover.nav_angles = angles

    dists, angles = nav_helper.to_polar_coords(x_obstacle, y_obstacle)
    Rover.obst_dists = dists
    Rover.obst_angles = angles

    update_beams_reading(Rover)

    return Rover