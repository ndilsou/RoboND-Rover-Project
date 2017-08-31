import numpy as np
import cv2
import image_processing as img_helper

ERR_ROLL = 1
ERR_PITCH = 0.25
ERR_YAW = 180
ROCK_RADIUS = 10
SCALE = 10
DELTA = 10

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160), invert=False):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    is_not_canvas = (img[:, :, 0] != 0) \
                    & (img[:, :, 1] != 0) \
                    & (img[:, :, 2] != 0)

    above_thresh = (img[:, :, 0] > rgb_thresh[0]) \
                   & (img[:, :, 1] > rgb_thresh[1]) \
                   & (img[:, :, 2] > rgb_thresh[2])

    if invert:
        above_thresh = ~above_thresh

    # Index the array of zeros with the boolean array and set to 1
    # we add a guard against modifying the black background.
    color_select[is_not_canvas & above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

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

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def find_rock(warped_rock, rock_radius):
    """
    returns the location of the rock as a circle with radius as defined in arguments.
    """
    # initialised empty
    xrock = np.array([])
    yrock = np.array([])

    y, x = warped_rock.nonzero()
    if y.any() and x.any():
        rock_idx = np.argmax(y)
        xrock, yrock = x[rock_idx], y[rock_idx]
        rock_circle = np.zeros_like(warped_rock)
        cv2.circle(rock_circle, (np.uint8(xrock), np.uint8(yrock)), rock_radius, (255, 255, 255), -1)
        xrock, yrock = rover_coords(rock_circle)

    return xrock, yrock

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
    warped_image = perspect_transform(img, src, dst)



    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    binary_rock = img_helper.rock_filter(img)
    binary_nav = color_thresh(warped_image)
    binary_obstacle = color_thresh(warped_image, invert=True)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = binary_obstacle*255
    Rover.vision_image[:, :, 1] = binary_rock*255
    Rover.vision_image[:, :, 2] = binary_nav*255
    # 5) Convert map image pixel values to rover-centric coords.replace(".",",")
    x_nav, y_nav = rover_coords(binary_nav)
    x_obstacle, y_obstacle = rover_coords(binary_obstacle)
    warped_rock = perspect_transform(binary_rock, src, dst)
    x_rock, y_rock = find_rock(warped_rock, ROCK_RADIUS)

    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_x_world, obstacle_y_world = pix_to_world(x_obstacle, y_obstacle, xpos, ypos, yaw, img_size, SCALE)
    nav_x_world, nav_y_world = pix_to_world(x_nav, y_nav, xpos, ypos, yaw, img_size, SCALE)
    rock_x_world, rock_y_world = pix_to_world(x_rock, y_rock, xpos, ypos, yaw, img_size, SCALE)


    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # update obstacle map
    if img_helper.is_valid_image(roll, pitch , yaw, eroll=ERR_ROLL, epitch=ERR_PITCH, eyaw=ERR_YAW):

        img_helper.delta_update(Rover.worldmap, obstacle_y_world, obstacle_x_world, 0, DELTA)

        # update navigation
        img_helper.delta_update(Rover.worldmap, nav_y_world, nav_x_world, 2, DELTA)
        # reduce certainty about obstacle.
        img_helper.delta_update(Rover.worldmap, nav_y_world, nav_x_world, 0, -DELTA * .66)

        # update rock sample map
        img_helper.delta_update(Rover.worldmap, rock_y_world, rock_x_world, 1, DELTA)
        # Rock area should be assumed navigable
        img_helper.delta_update(Rover.worldmap, rock_y_world, rock_x_world, 0, -DELTA)


    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    dists, angles = to_polar_coords(x_nav, y_nav)
    Rover.nav_dists = dists
    Rover.nav_angles = angles
    
    return Rover