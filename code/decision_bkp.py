import numpy as np
import navigation
from navigation import rad_to_deg, deg_to_rad, to_polar_coords
import image_processing
from config import Constants



# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        fwd_beam = Rover.beam_points[180]
        print(Rover.beam_points)
        fwd_dist, fwd_angle = to_polar_coords(fwd_beam[0], fwd_beam[1])
        print("fwd distance: {}".format(fwd_dist))
        close_dist, close_angle = Rover.closest_obstacle
        far_dist, far_angle = Rover.furthest_obstacle
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if Rover.seeing_sample:
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                mean_angle = np.mean(rad_to_deg(Rover.nav_angles))
                _, radian = to_polar_coords(*Rover.sample_pos)
                angle = rad_to_deg(radian)
                Rover.steer = np.clip(angle, -15, 15)

            elif len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                mean_angle = np.mean(rad_to_deg(Rover.nav_angles))

                print("wall angle", Rover.wall_angle)
                if np.isnan(Rover.wall_angle):
                    angle = mean_angle
                else:
                    wall_angle = rad_to_deg(Rover.wall_angle)
                    obst_dist, obst_rad = Rover.closest_obstacle
                    obst_angle = rad_to_deg(obst_rad)
                    angle = (2 * wall_angle + mean_angle) / 2

                Rover.steer = np.clip(angle, -15, 15)
                print("steer", Rover.steer)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            # elif len(Rover.nav_angles) < Rover.stop_forward:
            elif fwd_dist < Rover.stop_forward:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if far_dist > Rover.go_forward:
                # if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    angle = rad_to_deg(far_angle)
                    Rover.steer = np.clip(angle, -15, 15)
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to average between mean angle and normal to wall
                    mean_angle = np.mean(rad_to_deg(Rover.nav_angles))

                    print("wall angle", Rover.wall_angle)
                    if np.isnan(Rover.wall_angle):
                        angle = mean_angle
                    else:
                        wall_angle = rad_to_deg(Rover.wall_angle)
                        angle = (2 * wall_angle + mean_angle) / 2

                    Rover.steer = np.clip(angle, -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    print(Rover.throttle, Rover.brake, Rover.steer)
    return Rover

actions = {}

def register(mode):
    def decorator(func):
        actions[mode] = func
        return func
    return decorator


def do_something(Rover):
    Rover.throttle = Rover.throttle_set
    Rover.steer = 0
    Rover.brake = 0
    Rover.mode = "exploring"


@register("exploring")
def explore(Rover):
    # first check if we're stuck.
    if Rover.displacement < Rover.stuck_displacement and Rover.throttle != 0:
        Rover.mode = "stuck"
    else:
        dist, angle = Rover.closest_obstacle
        if dist > Rover.stop_forward:
            # If mode is forward, navigable terrain looks good
            # and velocity is below max, then throttle
            if Rover.vel < Rover.max_vel:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set
            else:  # Else coast
                Rover.throttle = 0
            Rover.brake = 0
            # Set steering to average angle clipped to the range +/- 15
            mean_angle = np.mean(rad_to_deg(Rover.nav_angles))

            print("wall angle", Rover.wall_angle)
            if np.isnan(Rover.wall_angle):
                angle = mean_angle
            else:
                wall_angle = rad_to_deg(Rover.wall_angle)
                obst_dist, obst_rad = Rover.closest_obstacle
                # obst_angle = rad_to_deg(obst_rad)
                angle = (2 * wall_angle + mean_angle) / 2

            Rover.steer = np.clip(angle, -15, 15)
            Rover.mode = "exploring"
        else:
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'stopping'

@register("stopping")
def stop(Rover):

    if Rover.vel > 0.2:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        # If we're not moving (vel < 0.2) then do something else
    elif Rover.vel <= 0.2:
        # Now we're stopped and we have vision data to see if there's a path forward
        dist, angle = Rover.closest_obstacle
        if dist < Rover.go_forward:
            Rover.throttle = 0
            # Release the brake to allow turning
            Rover.brake = 0
            # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
            furthest_dist, furthest_angle = Rover.furthest_obstacle
            Rover.steer = np.clip(furthest_angle, -15, 15) #  -15  # Could be more clever here about which way to turn
        # If we're stopped but see sufficient navigable terrain in front then go!
        # dist, angle = Rover.furthest_obstacle
        if len(Rover.nav_angles) >= Rover.go_forward_view:
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            # Set steer to average between mean angle and normal to wall
            mean_angle = np.mean(rad_to_deg(Rover.nav_angles))

            print("wall angle", Rover.wall_angle)
            if np.isnan(Rover.wall_angle):
                angle = mean_angle
            else:
                wall_angle = rad_to_deg(Rover.wall_angle)
                angle = (2 * wall_angle + mean_angle) / 2
            Rover.steer = np.clip(angle, -15, 15)
            Rover.mode = 'exploring'


@register("stuck")
def stuck(Rover):
    Rover.throttle = Rover.reverse_set
    Rover.brake = 0
    Rover.steer = 0
    Rover.mode = 'stopping'


@register("chasing")
def chase(Rover):
    pass


@register("picking_up")
def pick_up(Rover):
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    else:
        Rover.mode = "exploring"


def _decision_step(Rover):
    print("MODE: {}".format(Rover.mode))
    print("disp: {}".format(Rover.displacement))
    action = actions.get(Rover.mode, do_something)

    action(Rover)
    return Rover