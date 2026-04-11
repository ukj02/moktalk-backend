"""Unit tests for helpers (no LiveKit import)."""

import pytest

import helpers


def test_get_metadata_value_empty():
    assert helpers.get_metadata_value({}, ["a"], default="x") == "x"
    assert helpers.get_metadata_value(None, ["a"], default="x") == "x"


def test_get_metadata_value_exact_key():
    meta = {"greetingInstructions": "hello", "other": 1}
    assert (
        helpers.get_metadata_value(meta, ["greetingInstructions"], default="")
        == "hello"
    )


def test_get_metadata_value_case_insensitive():
    meta = {"GreetingInstructions": "hi there"}
    assert (
        helpers.get_metadata_value(
            meta, ["greetingInstructions", "greeting_instructions"], default=""
        )
        == "hi there"
    )


def test_get_metadata_value_first_key_wins():
    meta = {"a": "first", "b": "second"}
    assert helpers.get_metadata_value(meta, ["a", "b"]) == "first"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("no", False),
    ],
)
def test_debug_audio_enabled(monkeypatch, raw, expected):
    monkeypatch.setenv("AGENT_DEBUG_AUDIO", raw)
    assert helpers.is_debug_audio_enabled() is expected


def test_debug_audio_unset(monkeypatch):
    monkeypatch.delenv("AGENT_DEBUG_AUDIO", raising=False)
    assert helpers.is_debug_audio_enabled() is False
