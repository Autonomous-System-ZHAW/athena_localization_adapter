"""
Inputs for robot_localization: IMU in rad/s without bias, wheel speed v.

Subscribes `imu_raw` (sensor_msgs/Imu, in deg/s at the VESC) and `wheel_odom`
(nav_msgs/Odometry, only twist.linear.x); publishes `imu`
(sensor_msgs/Imu, yaw rate only) and `wheel_twist`
(geometry_msgs/TwistWithCovarianceStamped, longitudinal v and lateral = 0), both
with covariances in the frame `frame_id`. The computation is in `inputs.py`.
"""

from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from athena_localization_adapter.inputs import Inputs
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def seconds(stamp) -> float:
    """Time stamp of a message [s]."""
    return stamp.sec + stamp.nanosec * 1e-9


class InputsNode(Node):
    """Prepare IMU and wheel data, nothing else."""

    def __init__(self):
        super().__init__(
            "inputs_node", automatically_declare_parameters_from_overrides=True
        )
        params = {
            name: parameter.value
            for name, parameter in self.get_parameters_by_prefix("").items()
        }
        self.frame_id = str(params.pop("frame_id", "base_link"))
        params.pop("use_sim_time", None)
        self.inputs = Inputs.from_params(params)
        self.imu_pub = self.create_publisher(Imu, "imu", 10)
        self.twist_pub = self.create_publisher(
            TwistWithCovarianceStamped, "wheel_twist", 10
        )
        self.create_subscription(Imu, "imu_raw", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, "wheel_odom", self.on_wheel, qos_profile_sensor_data
        )
        self.get_logger().info(f"inputs_node started: {self.inputs}")

    def on_imu(self, raw: Imu):
        rate = raw.angular_velocity
        x, y, z = self.inputs.rates(seconds(raw.header.stamp), (rate.x, rate.y, rate.z))
        msg = Imu()
        msg.header.stamp = raw.header.stamp
        msg.header.frame_id = self.frame_id
        msg.angular_velocity.x, msg.angular_velocity.y = x, y
        msg.angular_velocity.z = z
        variance = self.inputs.rate_variance
        msg.angular_velocity_covariance = [
            variance,
            0.0,
            0.0,
            0.0,
            variance,
            0.0,
            0.0,
            0.0,
            variance,
        ]
        # -1: orientation and acceleration are not provided (sensor_msgs/Imu).
        msg.orientation_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0
        self.imu_pub.publish(msg)

    def on_wheel(self, odom: Odometry):
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = odom.header.stamp
        msg.header.frame_id = self.frame_id
        msg.twist.twist.linear.x = self.inputs.speed(
            seconds(odom.header.stamp), odom.twist.twist.linear.x
        )
        covariance = [0.0] * 36
        covariance[0] = self.inputs.speed_variance
        covariance[7] = self.inputs.lateral_variance
        msg.twist.covariance = covariance
        self.twist_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InputsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
