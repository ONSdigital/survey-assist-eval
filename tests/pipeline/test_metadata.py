"""Unit tests for shared evaluation pipeline helper utilities.

These tests focus on the helper functions in `survey_assist_eval.pipeline.metadata`.
"""

# pylint: disable=protected-access,missing-function-docstring

from argparse import Namespace
from pathlib import Path

import pytest

from survey_assist_eval.pipeline.metadata import update_metadata_with_args_and_defaults


def _args(
    *, output_shortname: str, input_file: str, batch_size: int | None, second_run: bool
) -> Namespace:
    return Namespace(
        output_shortname=output_shortname,
        input_file=input_file,
        batch_size=batch_size,
        second_run=second_run,
    )


@pytest.mark.utils
def test_update_metadata_sets_original_dataset_name_only_for_stg1(tmp_path: Path):
    args_stg1 = _args(
        output_shortname="STG1",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    out = update_metadata_with_args_and_defaults(args_stg1, {})
    assert out["original_dataset_name"] == str(tmp_path / "in.csv")

    args_second_run = _args(
        output_shortname="STG1",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=True,
    )
    out2 = update_metadata_with_args_and_defaults(args_second_run, {})
    assert "original_dataset_name" not in out2

    args_not_stg1 = _args(
        output_shortname="STG3",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    out3 = update_metadata_with_args_and_defaults(args_not_stg1, {})
    assert "original_dataset_name" not in out3


@pytest.mark.utils
def test_update_metadata_warns_and_overwrites_original_dataset_name_on_stg1_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    args = _args(
        output_shortname="STG1",
        input_file=str(tmp_path / "new.csv"),
        batch_size=None,
        second_run=False,
    )
    in_metadata = {"original_dataset_name": str(tmp_path / "old.csv")}

    out = update_metadata_with_args_and_defaults(args, in_metadata)
    captured = capsys.readouterr()

    assert "Warning: The original dataset name" in captured.out
    assert out["original_dataset_name"] == str(tmp_path / "new.csv")


@pytest.mark.utils
def test_update_metadata_warns_and_overwrites_batch_size_on_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    args = _args(
        output_shortname="STG2",
        input_file=str(tmp_path / "in.csv"),
        batch_size=20,
        second_run=False,
    )
    in_metadata = {"batch_size": 7}

    out = update_metadata_with_args_and_defaults(args, in_metadata)
    captured = capsys.readouterr()

    assert "Warning: The batch size in the input metadata" in captured.out
    assert out["batch_size"] == 20


@pytest.mark.utils
def test_update_metadata_keeps_batch_size_when_args_none_and_metadata_present(
    tmp_path: Path,
):
    args = _args(
        output_shortname="STG2",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    in_metadata = {"batch_size": 7}

    out = update_metadata_with_args_and_defaults(args, in_metadata)
    assert out["batch_size"] == 7


@pytest.mark.utils
def test_update_metadata_sets_batch_size_from_default_when_missing_and_args_none(
    tmp_path: Path,
):
    args = _args(
        output_shortname="STG2",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )

    out = update_metadata_with_args_and_defaults(args, {})
    assert out["batch_size"] == 100


@pytest.mark.utils
def test_update_metadata_sets_batch_size_from_args_when_missing(tmp_path: Path):
    args = _args(
        output_shortname="STG2",
        input_file=str(tmp_path / "in.csv"),
        batch_size=12,
        second_run=False,
    )

    out = update_metadata_with_args_and_defaults(args, {})
    assert out["batch_size"] == 12


@pytest.mark.utils
def test_update_metadata_fills_defaults_for_missing_keys(tmp_path: Path):
    args = _args(
        output_shortname="STG2",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    out = update_metadata_with_args_and_defaults(args, {"batch_size": 7})

    assert out["embedding_model_name"]
    assert out["llm_model_name"]
    assert out["sic_code_digits"] == 5


@pytest.mark.utils
def test_update_metadata_adds_stage_file_and_timing_keys(tmp_path: Path):
    args = _args(
        output_shortname="STG2",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    out = update_metadata_with_args_and_defaults(args, {})

    assert out["STG2_input_file"] == str(tmp_path / "in.csv")
    assert isinstance(out["STG2_start_timestamp"], float)
    assert isinstance(out["STG2_start_time_readable"], str)
