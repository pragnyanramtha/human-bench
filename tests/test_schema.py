import pytest
from pydantic import ValidationError

from humanbench.schema import Persona, Slot, TaskCard, load_personas


def test_persona_pool_loads_and_is_valid():
    personas = load_personas()
    assert len(personas) == 64
    assert any(not p.native for p in personas)


def test_l2_persona_needs_first_language():
    with pytest.raises(ValidationError):
        Persona(id="x", role="r", age=30, english="L2", lives="here", writing_level="everyday")


def test_slot_rejects_band_not_allowed_for_channel():
    with pytest.raises(ValidationError):
        Slot(id="chat-x", split="test", channel="chat", mode="thread_reply", tone="casual",
             audience="friend", stakes="routine", band="extended", persona_id="sre")


def _card(**overrides):
    persona = next(p for p in load_personas() if p.id == "leeds-plumber")
    base = dict(id="email-x", split="test", channel="email", mode="agent", tone="angry",
                audience="business", stakes="tense", band="short", persona_id=persona.id,
                persona=persona, prompt="They've sent the wrong valves again. Sort it.",
                environment="env/fixture.json", tools=["search", "read_item", "create_draft"],
                agent_checks={"delivery": "draft", "required": [{"target": "orders@valveco.example"}]})
    base.update(overrides)
    return TaskCard(**base)


def test_agent_card_valid():
    assert _card().mode == "agent"


def test_agent_card_needs_environment():
    with pytest.raises(ValidationError):
        _card(environment=None)


def test_ghostwrite_card_needs_user_voice():
    with pytest.raises(ValidationError):
        _card(mode="ghostwrite")


def test_agent_card_needs_agent_checks():
    with pytest.raises(ValidationError):
        _card(agent_checks=None)


def test_agent_card_rejects_unknown_tool():
    with pytest.raises(ValidationError, match="unknown tools"):
        _card(tools=["read_email"])


def test_direct_card_rejects_tools():
    with pytest.raises(ValidationError):
        _card(mode="direct", environment=None)
