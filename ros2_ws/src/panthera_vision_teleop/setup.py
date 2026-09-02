"""ament_python installation metadata for the teleoperation nodes."""

from glob import glob
import os

from setuptools import find_packages, setup


package_name = "panthera_vision_teleop"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="assaneko",
    maintainer_email="assaneko@example.com",
    description="Monocular-vision teleoperation demo for Panthera-HT.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "manual_target_node = panthera_vision_teleop.manual_target_node:main",
            "vision_bridge_node = panthera_vision_teleop.vision_bridge_node:main",
            "teleop_mapper_node = panthera_vision_teleop.teleop_mapper_node:main",
            "debug_pose_publisher = panthera_vision_teleop.debug_pose_publisher:main",
            "mock_trajectory_probe = panthera_vision_teleop.mock_trajectory_probe:main",
            "hardware_trajectory_probe = panthera_vision_teleop.hardware_trajectory_probe:main",
            "cartesian_trajectory_backend = panthera_vision_teleop.cartesian_trajectory_backend:main",
            "continuous_cartesian_backend = panthera_vision_teleop.continuous_cartesian_backend:main",
            "joint_velocity_cap_node = panthera_vision_teleop.joint_velocity_cap_node:main",
            "joint_trajectory_move = panthera_vision_teleop.joint_trajectory_move:main",
            "mock_cartesian_probe = panthera_vision_teleop.mock_cartesian_probe:main",
            "mock_continuous_cartesian_probe = panthera_vision_teleop.mock_continuous_cartesian_probe:main",
            "mock_scheduler_probe = panthera_vision_teleop.mock_scheduler_probe:main",
        ],
    },
)
