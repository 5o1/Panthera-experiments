"""Explicitly armed Panthera ros2_control experiment; never starts vision."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


CONFIRMATION = "ENABLE-PANTHERA-ROS2-CONTROL"


def _validate(context):
    if LaunchConfiguration("confirmation").perform(context) != CONFIRMATION:
        raise RuntimeError(
            "Real hardware launch blocked. Use the staged experiment script "
            f"and type {CONFIRMATION} only with the physical E-stop ready."
        )
    return []


def generate_launch_description() -> LaunchDescription:
    config_share = FindPackageShare("panthera_ht_config")
    teleop_share = FindPackageShare("panthera_vision_teleop")
    config_file = LaunchConfiguration("config_file")
    robot_xacro = PathJoinSubstitution([config_share, "config", "panthera_ht_hardware.urdf.xacro"])
    controllers = PathJoinSubstitution([teleop_share, "config", "continuous_control_hardware.yaml"])
    robot_description = {
        "robot_description": ParameterValue(
            Command([
                FindExecutable(name="xacro"), " ", robot_xacro,
                " config_file:=", config_file,
                " control_mode:=position_velocity",
            ]),
            value_type=str,
        )
    }
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="screen",
    )
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controllers, robot_description],
        output="screen",
    )
    state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    velocity_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_velocity_limit_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )
    velocity_cap = Node(
        package="panthera_vision_teleop",
        executable="joint_velocity_cap_node",
        parameters=[{
            "velocity_cap_radps": 0.6,
            "hardware_limit_radps": 1.0,
        }],
        output="screen",
    )
    return LaunchDescription([
        DeclareLaunchArgument("confirmation", default_value="BLOCKED"),
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution([config_share, "robot_param", "Follower_absolute.yaml"]),
        ),
        OpaqueFunction(function=_validate),
        state_publisher,
        controller_manager,
        state_spawner,
        RegisterEventHandler(OnProcessExit(target_action=state_spawner, on_exit=[arm_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=arm_spawner, on_exit=[velocity_spawner])),
        velocity_cap,
    ])
