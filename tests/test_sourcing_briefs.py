import re

import pytest

from humanbench.schema import NotesPack, Persona, TaskCard, load_personas
from humanbench.sourcing.briefs import (
    BriefLintError,
    lint_brief,
    render_brief,
    render_thread,
)
from humanbench.sourcing.screener import screener_for

PERSONAS = {p.id: p for p in load_personas()}


def card(**overrides) -> TaskCard:
    persona = PERSONAS[overrides.pop("persona_id", "maths-phd")]
    base = dict(
        id="forum-07", split="test", channel="forum", mode="thread_reply", tone="technical",
        audience="stranger", stakes="routine", band="medium", persona_id=persona.id, persona=persona,
        prompt="Someone on the maths forum still doesn't get why 0.999... = 1 after three replies. "
               "Help me write a reply that actually lands. Thread attached.",
        context="threads/forum-07.json",
        notes_pack=NotesPack(facts=["they tried the 1/3 argument and rejected it"],
                             wants=["no condescension", "one argument they can check themselves"]),
        must_include=[["limit", "limits"]],
        must_not_include=["obviously"],
    )
    base.update(overrides)
    return TaskCard(**base)


def test_brief_carries_the_same_raw_material():
    md = render_brief(card(), context_text="**asker**\n\nI still don't buy it.")
    assert "0.999... = 1" in md
    assert "they tried the 1/3 argument and rejected it" in md
    assert "no condescension" in md
    assert "I still don't buy it." in md
    assert "maths PhD student in number theory" in md and "Pune, India" in md and "27" in md
    assert "about 400-800 words" in md and "360 to 880" in md
    # models never see the requirement lists, so writers don't either
    assert "limit (or limits)" not in md and "obviously" not in md
    shown = render_brief(card(), context_text="x", include_requirements=True)
    assert "limit (or limits)" in shown and "obviously" in shown
    assert "headings" in md  # forum format rule is stated


def test_brief_length_range_never_crosses_the_floor():
    md = render_brief(card(persona_id="truck-driver", channel="email", mode="direct", audience="business",
                           band="short", context=None, tone="angry"))
    assert "Anything from 250 to 440" in md


def test_brief_never_mentions_ai_detection_or_writing_level():
    for c in (card(), card(persona_id="truck-driver", channel="email", mode="direct", audience="business",
                           band="short", context=None, tone="angry")):
        md = render_brief(c)
        assert not re.search(r"\b(AI|detector|detection|human-?like|sound human|benchmark|robot)\b", md, re.I)
        assert c.persona.writing_level.value not in md.lower().split()
        assert "writing level" not in md.lower()


def test_l2_writer_brief_states_first_language_neutrally():
    md = render_brief(card(persona_id="truck-driver", channel="email", mode="direct", audience="business",
                           band="short", context=None, tone="angry"))
    assert "First language: Polish" in md


def test_agent_brief_describes_the_environment_not_the_answers():
    c = card(id="email-agent-11", persona_id="leeds-plumber", channel="email", mode="agent", tone="angry",
             audience="business", stakes="tense", band="short", context=None, environment="envs/inbox-11/",
             tools=["search", "read_item", "create_draft"],
             agent_checks={"delivery": "draft", "required": [{"target": "orders@valve-supplier.example"}]},
             prompt="They've sent the wrong valves again. Sort it. Draft replies to whoever needs one.")
    c = c.model_copy(update={"must_include": ["third wrong delivery"], "must_not_include": ["bank"]})
    md = render_brief(c)
    assert "practice inbox" in md and "envs/inbox-11" not in md
    assert "in total across your drafts" in md
    # the writer's own selection is the human baseline, so the answer key stays hidden
    assert "third wrong delivery" not in md and "bank" not in md
    assert "third wrong delivery" in render_brief(c, include_requirements=True)


def test_lint_catches_performance_requests_in_card_content():
    with pytest.raises(BriefLintError):
        render_brief(card(prompt="Write a forum reply that doesn't sound like AI and passes as human please."))
    with pytest.raises(BriefLintError):
        lint_brief("Please write naturally so the detector can't tell.")
    lint_brief("Write it the way you would if you were really sending it.", "our new AI feature launches Monday")


def test_render_thread_json():
    out = render_thread({"messages": [{"from": "asker", "body": "Why?"}, {"from": "helper", "body": "Because."}]})
    assert out == "**asker**\n\nWhy?\n\n---\n\n**helper**\n\nBecause."
    thread = {"kind": "forum", "title": "0.999... = 1?", "user": "pk27", "reply_to": "m2", "messages": [
        {"id": "m1", "author": "pk27", "date": "2026-10-01", "body": "It's a limit."},
        {"id": "m2", "author": "newbie", "date": "2026-10-02", "body": "Still don't buy it."}]}
    out = render_thread(thread)
    assert out.startswith("**Thread: 0.999... = 1?**")
    assert "**pk27** (you) (2026-10-01)" in out
    assert "**newbie** (2026-10-02), the message you're replying to" in out


def test_screener_for_every_persona():
    for p in PERSONAS.values():
        spec = screener_for(p)
        assert spec.age_min <= p.age <= spec.age_max
        assert any(c.field == "occupation" for c in spec.criteria)
        assert any(c.field == "writing_sample" for c in spec.criteria)


def test_screener_l2_and_variety_rules():
    l2 = screener_for(PERSONAS["truck-driver"])
    assert l2.first_language == ["Polish"] and l2.english_variety is None
    assert l2.current_country == ["United Kingdom"]
    nigerian = screener_for(PERSONAS["biology-fresher"])
    assert nigerian.first_language == [] and nigerian.grew_up_in == ["Nigeria"]
    us = screener_for(PERSONAS["realtor"])
    assert us.first_language == ["English"] and us.current_country == ["United States"]


def test_no_persona_in_the_pool_is_a_minor():
    assert [p.id for p in PERSONAS.values() if screener_for(p).involves_minor] == []


def test_screener_flags_minors():
    teen = screener_for(Persona(id="minor", role="sixth-form student", age=16, english="UK",
                                lives="Leeds, England", writing_level="rough"))
    assert teen.involves_minor
    age = next(c for c in teen.criteria if c.field == "age")
    assert age.value[0] >= 18
    assert any("guardian" in n for n in teen.notes)
    assert not screener_for(PERSONAS["sre"]).involves_minor
