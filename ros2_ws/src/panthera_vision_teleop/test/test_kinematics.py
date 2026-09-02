from pathlib import Path

import numpy as np

from panthera_vision_teleop.kinematics import SerialChain


URDF = Path("/home/assaneko/Panthera_HT_ROS2/install/panthera_ht_ros_description/share/panthera_ht_ros_description/urdf/panthera_ht_ros_description.urdf")


def test_official_urdf_chain_has_expected_joint_order_and_limits():
    chain = SerialChain.from_urdf(URDF, "base_link", "link6")
    assert chain.joint_names == tuple(f"joint{index}" for index in range(1, 7))
    assert np.allclose(chain.lower, [-2.4, 0.0, 0.0, -1.6, -1.7, -2.5])
    assert np.allclose(chain.upper, [2.4, 3.2, 4.0, 1.6, 1.7, 2.5])


def test_geometric_jacobian_matches_finite_difference_translation():
    chain = SerialChain.from_urdf(URDF, "base_link", "link6")
    q = np.array([0.1, 0.5, 0.7, -0.2, 0.1, -0.1])
    transform, jacobian, _ = chain.forward_with_jacobian(q)
    epsilon = 1e-7
    for index in range(6):
        perturbed = q.copy()
        perturbed[index] += epsilon
        numerical = (chain.forward(perturbed)[:3, 3] - transform[:3, 3]) / epsilon
        assert np.allclose(numerical, jacobian[:3, index], atol=1e-6)


def test_inverse_recovers_nearby_official_chain_pose():
    chain = SerialChain.from_urdf(URDF, "base_link", "link6")
    seed = np.array([0.1, 0.5, 0.7, -0.2, 0.1, -0.1])
    expected = seed + np.array([0.01, 0.01, -0.01, 0.005, -0.005, 0.005])
    solution = chain.inverse(chain.forward(expected), seed)
    assert np.allclose(chain.forward(solution), chain.forward(expected), atol=1e-3)
