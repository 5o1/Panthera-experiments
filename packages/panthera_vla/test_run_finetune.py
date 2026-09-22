"""The checkpoint contract must actually be written.

``checkpoint.py`` was written during the 2026-09-19 restructure and had no
caller: every run produced weights that could not say what action chunk,
unnorm key or image preparation they were trained under, and evaluation fell
back to its own defaults.  These tests pin the wiring, not the file format --
``test_checkpoint.py`` covers the format.
"""

import pytest

from run_finetune import _flag, _options, _source_dataset, _validate_action_contract


def test_options_reads_key_value_pairs():
    assert _options(["--batch_size", "6", "--learning_rate", "5e-4"]) == {
        "batch_size": "6",
        "learning_rate": "5e-4",
    }


def test_options_treats_a_bare_flag_as_true():
    # draccus accepts both forms, and the contract must read them the same way.
    assert _options(["--use_proprio", "--batch_size", "6"]) == {
        "use_proprio": "true",
        "batch_size": "6",
    }


def test_options_keeps_negative_numbers_as_values():
    assert _options(["--learning_rate", "-1"]) == {"learning_rate": "-1"}


@pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
def test_flag_accepts_the_truthy_spellings(raw):
    assert _flag({"image_aug": raw}, "image_aug", False) is True


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off"])
def test_flag_accepts_the_falsy_spellings(raw):
    assert _flag({"image_aug": raw}, "image_aug", True) is False


def test_flag_falls_back_to_the_declared_default():
    assert _flag({}, "use_proprio", True) is True
    assert _flag({}, "use_diffusion", False) is False


def test_missing_source_dataset_stops_the_run(monkeypatch):
    # Failing before training is the point: a checkpoint that cannot name its
    # dataset is not reproducible, and discovering that afterwards costs a run.
    monkeypatch.delenv("PANTHERA_SOURCE_DATASET_ROOT", raising=False)
    with pytest.raises(SystemExit, match="PANTHERA_SOURCE_DATASET_ROOT"):
        _source_dataset({})


class _Dataset:
    name = "one-episode-subset"

    class contract:
        action_chunk = 25


def test_matching_action_contract_is_accepted():
    _validate_action_contract(
        model_dim=7, model_chunk=25, model_proprio=7,
        rlds_dim=7, rlds_chunk=25, rlds_proprio=7,
        dataset=_Dataset(),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rlds_chunk": 5}, "RLDS was registered with action chunk 5"),
        ({"model_dim": 8}, "ACTION_DIM=8"),
        ({"model_proprio": 28}, "PROPRIO_DIM=28"),
    ],
)
def test_mismatched_action_contract_stops_before_training(overrides, message):
    arguments = dict(
        model_dim=7, model_chunk=25, model_proprio=7,
        rlds_dim=7, rlds_chunk=25, rlds_proprio=7,
        dataset=_Dataset(),
    )
    arguments.update(overrides)
    with pytest.raises(SystemExit, match=message):
        _validate_action_contract(**arguments)
