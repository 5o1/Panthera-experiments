"""Launch the visual teleoperation pipeline; robot output defaults to off."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("panthera_vision_teleop")
    config = os.path.join(share, "config", "teleop.yaml")
    default_profile = os.path.join(share, "config", "teleop_safe.yaml")
    profile_config = LaunchConfiguration("profile_config")
    use_websocket = LaunchConfiguration("use_websocket")
    use_debug = LaunchConfiguration("use_debug")
    fake_robot = LaunchConfiguration("fake_robot_feedback")
    follow_fake_commands = LaunchConfiguration("fake_robot_follow_commands")
    robot_commands = LaunchConfiguration("publish_robot_commands")
    gripper_commands = LaunchConfiguration("publish_gripper_commands")
    continuous_backend = LaunchConfiguration("use_continuous_backend")

    return LaunchDescription([
        DeclareLaunchArgument("use_websocket", default_value="true"),
        DeclareLaunchArgument("use_debug", default_value="false"),
        DeclareLaunchArgument("fake_robot_feedback", default_value="false"),
        DeclareLaunchArgument("fake_robot_follow_commands", default_value="false"),
        DeclareLaunchArgument("debug_scenario", default_value="xyz_cycle"),
        DeclareLaunchArgument("fake_robot_speed_m_s", default_value="0.08"),
        DeclareLaunchArgument("publish_robot_commands", default_value="false"),
        DeclareLaunchArgument("publish_gripper_commands", default_value="false"),
        DeclareLaunchArgument("use_continuous_backend", default_value="false"),
        DeclareLaunchArgument(
            "profile_config",
            default_value=default_profile,
            description="YAML overlay for safe/normal mapping limits",
        ),
        Node(
            package="panthera_vision_teleop",
            executable="vision_bridge_node",
            name="vision_bridge_node",
            parameters=[config],
            condition=IfCondition(use_websocket),
            output="screen",
        ),
        Node(
            package="panthera_vision_teleop",
            executable="teleop_mapper_node",
            name="teleop_mapper_node",
            parameters=[config, profile_config, {
                "publish_robot_commands": ParameterValue(robot_commands, value_type=bool),
                "publish_gripper_commands": ParameterValue(gripper_commands, value_type=bool),
            }],
            output="screen",
        ),
        Node(
            package="panthera_vision_teleop",
            executable="debug_pose_publisher",
            name="debug_pose_publisher",
            parameters=[{
                "scenario": LaunchConfiguration("debug_scenario"),
                "publish_human": ParameterValue(use_debug, value_type=bool),
                "publish_robot_feedback": ParameterValue(fake_robot, value_type=bool),
                "follow_robot_commands": ParameterValue(follow_fake_commands, value_type=bool),
                "fake_robot_speed_m_s": ParameterValue(LaunchConfiguration("fake_robot_speed_m_s"), value_type=float),
            }],
            condition=IfCondition(PythonExpression(["'", use_debug, "' == 'true' or '", fake_robot, "' == 'true'"])),
            output="screen",
        ),
        Node(
            package="panthera_vision_teleop",
            executable="continuous_cartesian_backend",
            name="continuous_cartesian_backend",
            parameters=[{
                "enabled": True,
                "control_rate_hz": 50.0,
                "max_joint_velocity_radps": [0.6] * 6,
                "hardware_velocity_limit_radps": [1.0] * 6,
                "max_joint_acceleration_radps2": [2.0] * 6,
            }],
            condition=IfCondition(continuous_backend),
            output="screen",
        ),
    ])
