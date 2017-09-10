import numpy as np
from navigation import rad_to_deg, deg_to_rad, to_polar_coords


def mean_nav_angle(Rover):
    """
    Provide the mean angle from the field of vision, pixels are down-weighted if they've been
    visited before.
    """
    if Rover.nav_angles.any() and len(Rover.nav_angles) > 0:
        angle = np.average(rad_to_deg(Rover.nav_angles), weights=Rover.nav_weights)
    else:
        angle = -15
    return angle


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

    # if Rover.closest_obstacle:
    #     obst_dist, obst_angle = Rover.closest_obstacle
    #     print("!!! closest obstacle dist: {}, angle: {} !!!".format(obst_dist, obst_angle))
    #     obst_angle = rad_to_deg(obst_angle)

    return angle


def is_moving(Rover):
    """
    Checks that Rover has moved a certain distance since the last frame.
    """
    if Rover.pos is None or Rover.last_pos is None:
        return True

    displacement = ((Rover.pos[0] - Rover.last_pos[0]) ** 2 +
                    (Rover.pos[1] - Rover.last_pos[1]) ** 2) ** 1/2
    if displacement > Rover.etoll_disp:
        return True
    else:
        return False


def see_sample(Rover):
    """
    Checks if a Sample is in the current field of vision.
    """
    print("Seeing SAMPLE? {}".format(Rover.seeing_sample))
    # Are we seeing a sample on our left side?
    if Rover.seeing_sample:
        return True
    return False


def check_data(Rover):
    """
    Guard to prevent us from sending invalid data to the Rover.
    It looks like sending a n/a to the Rover breaks the connection.
    """
    if Rover.throttle == np.nan:
        Rover.throttle = 0
    if Rover.steer == np.nan:
        Rover.steer = 0
    if Rover.brake == np.nan:
        Rover.brake = 0


def track_sample(Rover):
    r_dists, r_rads = Rover.curr_sample_pos
    if Rover.near_sample:
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.brake = Rover.brake_set
        Rover.steer = 0
    else:
        if Rover.vel < Rover.max_pursuit_vel and is_moving(Rover):
            # Set throttle value to throttle setting
            Rover.throttle = Rover.rock_pursuit_set
        else:  # Else coast
            Rover.throttle = 0

        if np.mean(r_dists) < 0.01:
            # slow down
            Rover.brake = Rover.slow_down_set
        else:
            Rover.brake = 0
        angle = rad_to_deg(r_rads)
        Rover.steer = np.clip(np.mean(angle), -15, 15)


def forward_mode(Rover):
    # Check the extent of navigable terrain
    if len(Rover.nav_angles) >= Rover.stop_forward:
        # If mode is forward, navigable terrain looks good
        # and velocity is below max, then throttle
        if Rover.vel < Rover.max_vel:
            # Set throttle value to throttle setting
            Rover.throttle = Rover.throttle_set
        else:  # Else coast
            Rover.throttle = 0
        Rover.brake = 0
        # Set steering to average angle clipped to the range +/- 15
        Rover.steer = np.clip(set_angle(Rover), -15, 15)

    # If there's a lack of navigable terrain pixels then go to 'stop' mode
    elif len(Rover.nav_angles) < Rover.stop_forward:
        # Set mode to "stop" and hit the brakes!
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.mode = 'stop'


def stop_mode(Rover):
    # If we're in stop mode but still moving keep braking
    if Rover.vel > 0.2:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
    # If we're not moving (vel < 0.2) then do something else
    elif Rover.vel <= 0.2:
        # Now we're stopped and we have vision data to see if there's a path forward
        if len(Rover.nav_angles) < Rover.go_forward:
            Rover.throttle = 0
            # Release the brake to allow turning
            Rover.brake = 0
            # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
            Rover.steer = -15
        # If we're stopped but see sufficient navigable terrain in front then go!
        elif len(Rover.nav_angles) >= Rover.go_forward:
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            # Set steer to mean angle
            Rover.steer = np.clip(set_angle(Rover), -15, 15)
            Rover.mode = 'forward'

def stuck_mode(Rover):
    if Rover.vel > Rover.reverse_vel_set:
        Rover.throttle = Rover.reverse_set
        Rover.brake = 0
        Rover.steer = 0
    elif Rover.throttle < 0 and not is_moving(Rover):
        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = 0
        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
        Rover.steer = -15
        Rover.mode = "stop"
    else:
        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.mode = "stop"

# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):

    """
    The decision tree is broadly structured as follow:
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
    """
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # We're not moving but have positive velocity, we must be stuck.
        print("moving? {}, vel: {}".format(is_moving(Rover), Rover.vel))
        if (Rover.delta_timer.poll() and not is_moving(Rover)
            and (Rover.throttle != 0 or (Rover.throttle == 0 and Rover.steer != 0))):
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = "stuck"
        else:
            if see_sample(Rover):
                track_sample(Rover)
            else:
                # Check for Rover.mode status
                if Rover.mode == 'forward':
                    forward_mode(Rover)
                elif Rover.mode == 'stop':
                    stop_mode(Rover)
                elif Rover.mode == "stuck":
                    stuck_mode(Rover)

        # Last guard against erroneous data.
        check_data(Rover)

    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        Rover.mode = "stop"
        Rover.curr_curr_sample_pos = None

    return Rover
