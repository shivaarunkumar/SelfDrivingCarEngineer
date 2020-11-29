import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        #  Implement
        # Parse Inputs
        self.vehicle_mass = kwargs['vehicle_mass']
        self.wheel_radius = kwargs['wheel_radius']
        self.accel_limit = kwargs['accel_limit']
        self.decel_limit = kwargs['decel_limit']

        # YawController
        self.wheel_base = kwargs['wheel_base']
        self.steer_ratio = kwargs['steer_ratio']
        self.max_lat_accel = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']

        # Instantiate the required controllers
        # Yaw Controller for Steering Angle
        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, .1, self.max_lat_accel,
                                            self.max_steer_angle)

        # PID Controller for throttle
        kp = 0.3
        kd = 0.1
        ki = 0.
        mn = 0.  # Minimum Allowed Throttle
        mx = 0.2  # Maximum Allowed Throttle

        self.pid_controller = PID(kp, ki, kd, mn, mx)

        # Low Pass Controller to smooth our velocity (like we do for IMU sensors)
        tau = 0.5  # Cut off freq
        ts = .02  # Sample Time 50Hz
        self.lpf = LowPassFilter(tau, ts)

        # Cache time for delta_t
        self.last_time = rospy.get_time()

    def control(self, is_dbw_enabled, curr_vel, linear_vel, angular_vel):
        # Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # return 1., 0., 0.

        if not is_dbw_enabled:
            # Reset controllers
            self.pid_controller.reset()
            return 0., 0., 0.

        throttle = 1.
        brake = 0.
        steering = 0.

        curr_vel = self.lpf.filt(curr_vel)
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, curr_vel)

        vel_err = linear_vel - curr_vel
        curr_time = rospy.get_time()
        delta_t = curr_time - self.last_time
        self.last_time = curr_time

        throttle = self.pid_controller.step(vel_err, delta_t)

        if linear_vel == 0 and curr_vel < 0.1:
            throttle = 0
            brake = 700
        elif throttle < .1 and vel_err < 0:
            throttle = 0.
            decel = max(vel_err, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
