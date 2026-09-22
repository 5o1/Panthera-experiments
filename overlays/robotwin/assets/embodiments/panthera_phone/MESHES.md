# meshes/ is not in the repository

The mesh set is 36 MB of binary copied from the pinned official ROS 2 model, and
`build_panthera_embodiment.py` regenerates it deterministically from that source
together with `provenance.json`, which records the input hashes.  Keeping the
text here and the binaries out is what lets a fresh checkout verify the
embodiment is the expected one without carrying the meshes in git history.

Rebuild with:

    python packages/panthera_sim/build_panthera_embodiment.py \
      --ros2-root ~/Panthera_HT_ROS2 \
      --output-root overlays/robotwin/assets/embodiments/panthera_phone
