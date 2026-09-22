# Superseded implementations

Replaced by `packages/panthera_sim`, which merged them into one dataset access
layer and one executor. They are kept only because the recorded baselines were
produced by them: the expert replay 1249/1280 and the policy rollout 0/12 were
reproduced episode-for-episode on the replacement before these were retired, and
these files are what that comparison was against.

They read `scene_info.json` directly and rebuild scenes from seeds, which is the
defect the replacement exists to remove. Do not extend them.
