"""The subset must retain metadata needed for deterministic evaluation."""

from build_episode_subset import _episode_records


class _Dataset:
    def episode_record(self, episode_id):
        return {
            "episode_seed": 10_000_000 + episode_id,
            "task_randomization": {"camera_randomization": {"pose_mode": "fixed"}},
            "camera_randomization": {"position_xyz_m": [0.0, 0.4, 1.3]},
        }


def test_subset_keeps_only_selected_camera_records():
    records = _episode_records(_Dataset(), [2, 7])
    assert sorted(records) == ["episode_2", "episode_7"]
    assert records["episode_2"]["panthera_episode"]["episode_seed"] == 10_000_002
    assert (
        records["episode_7"]["panthera_episode"]["camera_randomization"]
        ["position_xyz_m"]
        == [0.0, 0.4, 1.3]
    )
