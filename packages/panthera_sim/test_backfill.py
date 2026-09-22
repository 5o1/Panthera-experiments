"""What a backfilled snapshot may and may not claim about its assets."""

from backfill import _assets


def test_symmetric_scene_profile_names_the_embodiment():
    assets = _assets({"scene_profile":
                      "panthera_phone_symmetric_single_grasp_"
                      "direct_release_cylinder_socket_v2"})
    assert assets["embodiment_profile"] == "panthera_phone_symmetric"
    assert assets["scene_profile"].startswith("panthera_phone_symmetric")


def test_pre_symmetric_collection_asserts_no_embodiment():
    # The schema 5 pilot recorded this, and it predates the symmetric
    # embodiment. Naming that embodiment would describe the dataset as
    # something it is not.
    assets = _assets({"scene_profile":
                      "panthera_phone_randomized_cylinder_socket_v2"})
    assert assets["embodiment_profile"] is None
    assert assets["scene_profile"] == "panthera_phone_randomized_cylinder_socket_v2"


def test_missing_scene_profile_claims_nothing():
    assets = _assets({})
    assert assets["scene_profile"] is None
    assert assets["embodiment_profile"] is None
