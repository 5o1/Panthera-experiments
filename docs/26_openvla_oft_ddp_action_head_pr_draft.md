# OpenVLA-OFT PR Draft: Synchronize Continuous Action Heads under DDP

## Proposed title

Fix DDP synchronization for continuous action heads

## Proposed PR body

### Summary

Fixes [#160](https://github.com/moojink/openvla-oft/issues/160).

The continuous L1 and diffusion action heads are wrapped in
`DistributedDataParallel`, but the training path currently calls methods on
`action_head.module` directly. That bypasses `DistributedDataParallel.forward()`
and its per-iteration reducer preparation. With different local batches on each
rank, the action-head gradients are not reduced correctly and the replicas can
diverge after an optimizer step.

This PR:

- adds standard `forward()` methods to `L1RegressionActionHead` and
  `DiffusionActionHead`;
- routes the parameterized L1 and diffusion training predictions through the
  DDP wrapper;
- adds a two-rank CPU/Gloo regression test for both continuous action heads;
- keeps the existing `predict_action()` and `predict_noise()` methods for
  compatibility.

### Root cause

`init_module()` returns a DDP-wrapped action head, but `run_forward_pass()`
previously bypassed that wrapper:

```python
predicted_actions = action_head.module.predict_action(actions_hidden_states)
noise_pred = action_head.module.predict_noise(actions_hidden_states)
```

Accessing `.module` is appropriate when code needs an implementation detail
that DDP does not expose, but a parameterized training forward pass must go
through the wrapper. Otherwise DDP does not prepare its reducer for that
iteration before autograd runs.

### Change

Both action heads now expose their existing prediction operation through
`nn.Module.forward()`:

```python
class L1RegressionActionHead(nn.Module):
    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)


class DiffusionActionHead(nn.Module):
    def forward(self, actions_hidden_states):
        return self.predict_noise(actions_hidden_states)
```

The training path calls the DDP wrapper itself:

```python
predicted_actions = action_head(actions_hidden_states)
noise_pred = action_head(actions_hidden_states)
```

The no-gradient reverse-diffusion sampling path is intentionally unchanged. It
still accesses `action_head.module` because it also needs the underlying noise
scheduler and time encoder, and it does not perform a parameterized backward
pass.

### Regression test

The new test starts two CPU processes with the Gloo backend. Each rank receives
a different local input and target, performs one AdamW step, and then exchanges
the complete action-head parameter vector. The test requires exact equality
between ranks for both:

- `L1RegressionActionHead`;
- `DiffusionActionHead`.

It also guards the training call site so a future change cannot silently switch
back to `action_head.module.predict_*()` for the parameterized forward pass.

The test uses Python's standard `unittest` runner and does not add a test-only
dependency:

```bash
python -m unittest -v tests.test_action_head_ddp
```

Observed result:

```text
test_action_head_loader_restores_sys_modules ... ok
test_training_source_calls_ddp_wrapper ... ok
test_two_rank_optimizer_step_keeps_action_heads_synchronized ... ok

Ran 3 tests
OK
```

The same tests also pass under pytest:

```text
3 passed, 2 subtests passed
```

The isolated loader snapshots and restores every synthetic `prismatic` entry
in `sys.modules` in a `finally` block. This keeps the lightweight test import
from affecting subsequent tests in the same Python process.

### Additional multi-GPU validation

I also reproduced the issue independently with three GPUs and different
rank-local batches. The table reports the maximum difference between ranks
after one AdamW step; the output column evaluates all replicas on the same
post-step input.

| Action head | Call path | Gradient max diff | Parameter max diff | Common-output max diff |
|---|---|---:|---:|---:|
| L1 | bypass DDP via `.module.predict_action()` | 0.083215743 | 0.001000002 | 0.171851695 |
| L1 | DDP `forward()` | 0 | 0 | 0 |
| Diffusion | bypass DDP via `.module.predict_noise()` | 0.130688921 | 0.001000002 | 0.098186493 |
| Diffusion | DDP `forward()` | 0 | 0 | 0 |

The negative controls still diverge, so the all-zero fixed results are not
caused by identical local batches or by a test that lost sensitivity.

### Compatibility and scope

- No checkpoint keys or tensor shapes change.
- The existing `predict_action()` and `predict_noise()` APIs remain available.
- Single-rank numerical operations are unchanged; only the DDP call boundary
  changes during training.
- Discrete action-token training is unaffected.
- This PR does not change learning rates, stopping rules, action dimensions,
  robot-platform constants, checkpoint selection, or evaluation behavior.
- The regression test targets action-head synchronization rather than running a
  complete VLA fine-tuning job.

## Branch information

```text
Repository: https://github.com/moojink/openvla-oft
Base:       upstream/main@e4287e94541f459edc4feabc4e181f537cd569a8
Branch:     fix/ddp-action-head-forward-sync
Fix commit: bf0b44c53cab576fe8c2e3d5f38dc06666e75528
Review fix: 2c767d928fbfdf57f100b8840b6e3b4a9d5e5559
```

The branch was pushed to `5o1/openvla-oft` and opened against upstream `main`:

<https://github.com/moojink/openvla-oft/pull/162>
