"""Launch a Panthera-shaped ros2_control graph with fake hardware only.

This launch file deliberately uses the official mock URDF and never accepts a
real-hardware switch or SDK configuration path.  Its sole purpose is testing
controller names, joint ordering and FollowJointTrajectory command flow.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_share = FindPackageShare("panthera_ht_config")
    teleop_share = FindPackageShare("panthera_vision_teleop")
    robot_xacro = PathJoinSubstitution(
        [config_share, "config", "panthera_ht_ros_description.urdf.xacro"]
    )
    initial_positions = PathJoinSubstitution(
        [config_share, "config", "initial_positions.yaml"]
    )
    controllers = PathJoinSubstitution(
        [teleop_share, "config", "continuous_control_mock.yaml"]
    )
    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    FindExecutable(name="xacro"),
                    " ",
                    robot_xacro,
                    " initial_positions_file:=",
                    initial_positions,
                    " use_mock_hardware:=true",
                ]
            ),
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
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )
    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    start_arm_after_state_broadcaster = RegisterEventHandler(
        OnProcessExit(target_action=joint_state_spawner, on_exit=[arm_spawner])
    )
    cartesian_backend = Node(
        package="panthera_vision_teleop",
        executable="cartesian_trajectory_backend",
        parameters=[{
            "enabled": True,
            "stale_timeout_sec": 0.5,
            "max_joint_step_rad": 0.01,
            "segment_duration_sec": 0.25,
        }],
        condition=IfCondition(LaunchConfiguration("use_cartesian_backend")),
        output="screen",
    )
    continuous_backend = Node(
        package="panthera_vision_teleop",
        executable="continuous_cartesian_backend",
        parameters=[{
            "enabled": True,
            "control_rate_hz": 50.0,
            "max_joint_velocity_radps": [0.6] * 6,
            "hardware_velocity_limit_radps": [1.0] * 6,
            "max_joint_acceleration_radps2": [2.0] * 6,
        }],
        condition=IfCondition(LaunchConfiguration("use_continuous_backend")),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_cartesian_backend", default_value="false"),
            DeclareLaunchArgument("use_continuous_backend", default_value="false"),
            state_publisher,
            controller_manager,
            joint_state_spawner,
            start_arm_after_state_broadcaster,
            cartesian_backend,
            continuous_backend,
        ]
    )
