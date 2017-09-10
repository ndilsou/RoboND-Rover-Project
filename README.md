## Project: Search and Sample Return
____

This implementation of the Search and Sample return project was ran on a Ubuntu 16.04 device with the following specifications:

    Memory: 7.7 GiB
    Processor: Intel Core i7-3537U CPU @ 2.00GHz x 4
    Graphics: Intel Ivybridge Mobile
    OS Type: 64-bit

Simulator was run in 1024x576 with Graphics Quality set to Good.

The objective of the project was to provide a robot with perception and decision capability. The Robot should be able to do a mapping of its environment and optionally collect up to 6 rock sample.


### Notebook Analysis
___

#### Landmarks detection procedure:

##### Navigable Terrain:

Navigable terrain is detected from the warped field of vision of the robot, by thresholding and binarizing the vision image to filter out any pixel with RBG < (160, 160, 160).

##### Obstacles:

To locate the non-navigable terrain we reverse the procedure for navigable terrain and filter out all pixels with RGB > (160, 160, 160). Care must be taken to not return the black background in the image as obstacle. To do that a flag was added to the __color_thresh(image, rgb_thresh, invert, no_canvas)__ function to filter out the "canvas".

The function is provided below:
```python
def color_thresh(image, rgb_thresh=(160, 160, 160), invert=False, no_canvas=True):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(image[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (image[:,:,0] > rgb_thresh[0]) \
            & (image[:,:,1] > rgb_thresh[1]) \
            & (image[:,:,2] > rgb_thresh[2])
    if invert:
        above_thresh = ~above_thresh
    if no_canvas:
        is_not_canvas =  (image[:,:,0] != 0) \
            & (image[:,:,1]!= 0) \
            & (image[:,:,2] != 0)
        above_thresh = is_not_canvas & above_thresh
    # Index the array of zeros with the boolean array and set to 1
    # we add a guard against modifying the black background.
    color_select[above_thresh] = 1
    return color_select
```

##### Samples:

locating samples involves the following steps.
1. filtering the raw RGB image. To do so, we first apply a Gaussian Blur to the image, then run a two band threshold in HSV to isolate the yellow color and we binarize the result. This is performed by the function __rock_filter(image, hsv_lower, hsv_upper, ksize, sigmaX)__. A large kernel and variance is recommended to prevent some brightly lit obstacles to be false positive.
2. apply the perspective transform to the resulted image.
3. get the position of the best X,Y pixels to represent the center of the rock. Generate a circle around this point and return those as the potential rock position. This is performed by the function __find_rock(warped_rock, rock_radius)__

#### Overview of process_image():

This method collects all the detection functionality described above to generate the navigation view and the map for the Rover. The first step is the identification of navigable terrain, samples and obstacles by following the procedures detailed in the previous section.

Once this is done, we use the telemetry of the Robot to check that it's Roll and pitch did not lead to poor readings from the camera. If Roll and Pitch are above a threshold of 1.5 degrees we do not send an update to the map.

The intensity of the pixels is updated gradually by increment of 10 with a cap at 255 and a floor at 0. This way as the Rover sees the same pixel repeatedly it increases it's confidence about the nature of the pixel.
To correct erroneous readings, we reduce the weight of a navigable pixel if it is identified as obstacle in the next reading. This allow for a self-correcting map that get cleaner as the Rover passes around the same areas. This is particularly useful to gain great certainty about the location of obstacles

The full procedure for map update is shown below:

```python
 if is_valide_image(roll, pitch , yaw, eroll=ERR_ROLL, epitch=ERR_PITCH):
        delta_update(data.worldmap, obstacle_y_world, obstacle_x_world, 0, DELTA)
        delta_update(data.worldmap, obstacle_y_world, obstacle_x_world, 2, -DELTA * 0.3)
        # update navigation
        delta_update(data.worldmap, nav_y_world, nav_x_world, 2, 2 * DELTA)
        # reduce certainty about obstacle.
        delta_update(data.worldmap, nav_y_world, nav_x_world, 0, -DELTA)
        # update rock sample map
        delta_update(data.worldmap, rock_y_world, rock_x_world, 1, 255)
```

A drawback of this procedure is that the Rover is always as risk of forgetting some pieces of navigable terrain if it is in a narrow pass. A possible solution would be to lower the delta of the down-weighting based on a pixel's age. Pixels that have been set a long time ago should not need to be updated too aggressively

#### Additional work done in the notebook: 2D Cloud Point for wall navigation.

In addition to the points developed above, a section of the notebook focuses on building a simple form of cloud point. This is used as part of the Rover navigation system to help it find the best steering angle to crawl along a wall.

A possible alternative usage of this feature would be obstacle avoidance, as we can identify the location of the obstacles around the Rover in its own coordinate frame, we should be able to devise a way for it to move away from then, akin to a potential field.

The relevant functions for this functionality are:

* __dominant_channel_filter(image, channel)__
* __world_to_pix(x_pix_world, y_pix_world, xpos, ypos, yaw,  scale)__
* __find_beam_points(worldmap, degrees, xpos, ypos, yaw, scale, atol)__

### Autonomous Navigation and Mapping
___
#### Structure of the project:

Several files were added to the project to unclutter the two files provided.
All the files and their content is described below:

- *image_processing.py* [new]: Provides a set of function to extract info from the image. most of the code for Rock and obstacle detection is stored there.

- *navigation.py* [new]: Provides a set of function to manipulate the map and vision with a focus on helping the Rover find its way. The code for the 2D point cloud is stored there.

- *perception.py* [from udacity]: Contains the code used to update the map and identify landmarks.

- *decision.py* [from udacity]: Provide the decision tree used by the Rover to navigate and the related functions.

- *drive_rover.py* [from udacity]: Provides Rover class and server to connect to the simulator. A new class was added to provide a stateful timer __DeltaTimer__. New attributes were added to the Rover to support its new functionalities.

- *supporting_function.py* [from udacity]: Provide functions to update the Rover with new telemetry and update the maps for the simulator.



#### Overview of the decision tree:

The decision tree is broadly structured as follow:
```
    [Has nav data?]
    Yes:
        [Is Stuck?]
         Yes:
            Mode = "stuck"
         No:
            [Sample in view?]
                 Yes:
                    move toward sample until close enough to pick it up.
                    Then transition to Mode = "stop" when sample has been picked up.
                No:
                    [Mode is forward?]
                        Do Forward
                    [Mode is stop?]
                        Do Stop
                    [Mode is stuck?]
                        Do Stuck

    No:
        Adopt Default behavior.
```

#### Perception Step

The main method is broken down in 5 steps. Let us start by giving an overview of all the logic deployed for the perception step.

To navigate the Rover needs more information than just knowing the amount of navigable view in front of it.
Some useful heuristics are:

* Pixels that have already been visited should be given less importance that pixels leading to unknown area.
* Following the left wall is usually a good idea.  A way to do that is to go in the direction of the vector parallel to the wall.

Using those two simple heuristics, plus the original nav_angles, we can look for a lot more information in the input image.

We augmented the Rover with a __visited_map__ attribute. It contains a binary representation of the map where any pixel that the Rover already drove on is set to True. This way we can follow the path of the Rover.

1. Mark the current location as visited in the __visited_map__ using __visit_location(Rover)__.
2. Extract landmarks from the image. cf. process_image for more details.
3. Update the map and vision image. cf. process_image for more details.
4. Downweight the pixels that have already been visited in the current warped field of vision using __weight_visited(Rover, nav_x_world, nav_y_world)__.
5. Perform a beam reading using __update_beams_reading(Rover)__ of the 2D point cloud between 45 degrees and 135 degrees up to 5m. This means we got all the obstacles within 5m on our left. Find the average point. Take the normal vector between the vector location (0,0) and this average point. This vector will be a good approximate of the parallel to the wall.

We provide below the code for the 2D point cloud mechanism:

```Python

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

```

![](./results/point_cloud.jpg?raw=true)

#### Decision Step

The overall structure of the tree used in the decision step is provided above. For this reason we will use this section to cover in more details some important features.

###### Selecting the angle.
If the Rover is in forward mode, we provide the steering by build an average angle from the wall normal vector and the navigable pixels in the field of vision. navigable pixels that were already visited are given a 5% weight while unvisited pixels have a weight of 1. Those weights are normalized before passing them to the __set_angle()__ function. in the final average, the wall normal vector is weighted at 0.33% vs 0.66% for the navigable angles.

This leads to interesting exploration pattern. If all the visible pixel are new, the Robot will go for this direction, if the Robot has to choose between an open field with lots of visited pixels or a more narrow path with new pixel it will tend to go there.

In addition to that, whenever enough readings can be made from the 2D point cloud, the robot will be biased toward driving along the wall.

```python
def set_angle(Rover):
    """
    Provides the best angle for the Rover, based on available informations:
    - Navigable pixels angles, weighted to focus the angle on unvisited pixels.
    - Direction of the Normal vector the wall. Allows the Rover to adopt a wall-crawling strategy.
    """
    mean_angle = mean_nav_angle(Rover)
    if not np.isnan(Rover.wall_point).all():
        wall_normal_vect = rad_to_deg(Rover.wall_angle)
        weights = np.array([0.33, 0.66])
        angles = np.array([wall_normal_vect, mean_angle])
        print(mean_nav_angle(Rover), wall_normal_vect)
        angle = np.average(angles, weights=weights)
        print("unclipped angle: {}".format(angle))
    else:
        angle = mean_angle

    return angle
```

###### Getting unstuck.
To assess if it is stuck, the Rover checks if it has been able to move a certain distance between two frames. If the Rover is stuck it attempts to drive backward. It then transition to a "stop" mode to reevalute the best way to go. If Going backward is not enough, it will brake and turn at -15 degrees.

###### Chasing rocks.
If a sample is identified in the field of vision, the Rover will set is steering angle to it. It will cap it's speed to a lower value and it will start slowing down when getting close enough. Finally it will completely brake if it is near the sample. This simple strategy has proved successful in almost all cases. Depending on the angle of the approach, or the speed while in Forward mode, it can happen that the Rover will miss.


#### Results and possible improvements:

![](./results/success_run2.jpg?raw=true)

This Rover design allowed us to reach 95% of the environment mapped at 75.6% fidelity with 6 samples collected most of the time.
We received updates at 8 FPS.
Current know issues are:
* Lack of clean exception management.
* If the Rover is reversing and hits the wall behind it, it may not be able to transition into the stopped mode and will keep driving backward forever.
* Under some circumstance the bright rocks in the starting area may be seen as samples. This will lead the Rover to hit the obstacle. It usually managed to find its way back afterward.
* The rocks in the middle area are still hard to go by.
* The Rover needs to be aware of its own size and of obstacle. A Solution could be to grow the obstacles in the field of vision.
    Maybe we could do it through a dilation or an erosion.
Possible improvement:

* Using the 2D point Cloud to build a proper potential field navigation system. Unvisited pixels would be part of the attractive field and we would use the existing information on the obstacle locations around the Rover to build the repulsive field.

* Graph of navigable positions: We could store the navigable locations as part of a graph and use it with A* to easily move back the starting point. Or even to move faster towards the edge of the graph, where more locations are left to be explored. This could work by having a special unexplored type of node to mark areas of interest.
