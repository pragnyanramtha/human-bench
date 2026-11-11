from humanbench.scoring import floor


def test_judge_never_scores_its_own_company():
    judges = ["api:claude-sonnet-5-5", "api:gpt-6.1-terra"]
    assert floor.eligible(judges, "gpt-6-astra") == ["api:claude-sonnet-5-5"]
    assert floor.eligible(judges, "claude-opus-5-5") == ["api:gpt-6.1-terra"]
    assert floor.eligible(judges + judges, "gemini-4-argon") == judges  # duplicates removed
    import pytest
    with pytest.raises(ValueError):
        floor.eligible(judges, "mystery-model-9")


def test_parse_needs_every_answer_as_a_boolean():
    ok = '{"coherent": true, "does_task": true, "facts": true, "tone": true, "persona": false, "reason": "too formal"}'
    assert floor.parse("Here you go:\n" + ok)["persona"] is False
    assert floor.parse('{"coherent": "yes", "does_task": true, "facts": true, "tone": true, "persona": true}') is None
    assert floor.parse("no json") is None


def test_verdict_majority_and_ties_fail_and_no_votes_is_unscored():
    yes = dict.fromkeys(floor.QUESTIONS, True)
    no_task = {**yes, "does_task": False}
    assert floor.verdict([yes, yes, no_task]) == (1.0, [])
    assert floor.verdict([yes, no_task]) == (0.0, ["floor:does_task"])
    assert floor.verdict([])[0] is None
    assert floor.verdict([yes], electorate=3)[0] is None  # a missing vote must not shrink the majority
    assert floor.verdict([yes, yes, no_task], electorate=3) == (1.0, [])


def test_privacy_answer_is_required_when_asked():
    base = '{"coherent": true, "does_task": true, "facts": true, "tone": true, "persona": true, "reason": "ok"}'
    assert floor.parse(base, privacy=True) is None
    with_p = base.replace('"reason"', '"private": false, "reason"')
    assert floor.parse(with_p, privacy=True)["private"] is False
    assert "private" not in floor.parse(with_p)
    yes = dict.fromkeys(floor.QUESTIONS, True)
    assert floor.verdict([{**yes, "private": False}]) == (0.0, ["floor:private"])


def test_company_of_openrouter_and_cli_names():
    assert floor.company("openrouter:google/gemini-4-argon") == "google"
    assert floor.company("api:claude-sonnet-5-5") == "anthropic"
    assert floor.company("api:gpt-6.1-terra") == "openai"
    judges = ["api:claude-sonnet-5-5", "openrouter:google/gemini-4-argon", "api:gpt-6.1-terra"]
    assert floor.eligible(judges, "gpt-6-luna") == judges[:2]
    assert floor.eligible(judges, "anthropic/claude-opus-5.5") == judges[1:]


def test_parse_rejects_a_question_answered_twice():
    assert floor.parse('{"coherent": false, "coherent": true, "does_task": true, "facts": true, "tone": true, "persona": true}') is None
