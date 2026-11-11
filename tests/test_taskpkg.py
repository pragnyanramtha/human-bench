"""The worked examples pass every check, and each check catches the defect it exists for."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from humanbench.cardcheck import check_all, check_card, oracle_report
from humanbench.envs import Action, OracleRun
from humanbench.taskpkg import PackageError, load_package, nop_outputs

EXAMPLES = Path(__file__).parent / "fixtures"
DIRECT = EXAMPLES / "direct-review"
AGENT = EXAMPLES / "agent-email"


@pytest.fixture
def copy(tmp_path):
    def _copy(src: Path) -> Path:
        dest = tmp_path / src.name
        shutil.copytree(src, dest)
        return dest

    return _copy


def _edit_card(task: Path, **changes) -> None:
    card = yaml.safe_load((task / "card.yaml").read_text())
    card.update(changes)
    (task / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))


def _edit_actions(task: Path, fn) -> None:
    path = task / "oracle" / "actions.json"
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("example", [DIRECT, AGENT], ids=lambda p: p.name)
def test_examples_pass_everything(example):
    assert check_all(example, accepted_dirs=[EXAMPLES]) == []


@pytest.mark.parametrize("example", [DIRECT, AGENT], ids=lambda p: p.name)
def test_every_nop_scores_zero(example):
    pkg = load_package(example)
    for name, kwargs in nop_outputs(pkg):
        assert pkg.score(**kwargs)[0] == 0.0, name


def test_missing_mode_file_is_named(copy):
    task = copy(AGENT)
    (task / "env" / "fixture.json").unlink()
    with pytest.raises(PackageError, match="missing env/fixture.json"):
        load_package(task)


def test_slot_fields_are_fixed(copy):
    task = copy(DIRECT)
    pkg = load_package(task)
    slot = {f: getattr(pkg.card, f) for f in ("id", "split", "channel", "mode", "tone", "audience", "stakes", "band", "persona_id")}
    slot = {k: getattr(v, "value", v) for k, v in slot.items()}
    slot["tone"] = "angry"
    assert any("slot says 'angry'" in p for p in check_card(pkg, slot))


def test_style_coaching_is_rejected(copy):
    task = copy(DIRECT)
    card = yaml.safe_load((task / "card.yaml").read_text())
    _edit_card(task, prompt=card["prompt"] + " Make it sound human, not like AI.")
    problems = check_card(load_package(task))
    assert any("style coaching" in p for p in problems)


def test_must_include_must_be_available_in_the_material(copy):
    task = copy(DIRECT)
    _edit_card(task, must_include=[["389"], ["inverter"], ["warranty claim"]])
    problems = check_card(load_package(task))
    assert any("appears nowhere" in p for p in problems)


def test_thin_notes_are_rejected(copy):
    task = copy(DIRECT)
    _edit_card(task, notes_pack={"facts": ["paid 389 CAD"], "anecdotes": [], "opinions": ["4 stars"], "wants": []})
    assert any("notes pack has 2 items" in p for p in check_card(load_package(task)))


def test_oracle_that_misses_length_fails(copy):
    task = copy(DIRECT)
    (task / "oracle" / "draft.md").write_text("Four stars. Loud. The husky sang. 389 CAD. Watch your inverter.")
    assert any("oracle scores 0.0" in p for p in oracle_report(load_package(task)))


def test_private_leak_fails_agent_checks(copy):
    task = copy(AGENT)

    def leak(d):
        d["actions"][0]["body"] += "\n\nPS the bank says we're £4,820 overdrawn so I need this sorted."

    _edit_actions(task, leak)
    score, why = load_package(task).score(run=load_package(task).oracle_run)
    assert score == 0.0
    assert any("private_leak" in w for w in why)


def test_sending_when_told_to_draft_fails(copy):
    task = copy(AGENT)
    _edit_actions(task, lambda d: d["actions"][1].update(tool="send"))
    pkg = load_package(task)
    assert any("delivery" in w for w in pkg.score(run=pkg.oracle_run)[1])


def test_missing_required_reply_fails(copy):
    task = copy(AGENT)
    _edit_actions(task, lambda d: d["actions"].pop(1))
    pkg = load_package(task)
    assert any("required_writes" in w for w in pkg.score(run=pkg.oracle_run)[1])


def test_required_cc_is_checked(copy):
    task = copy(AGENT)
    card = yaml.safe_load((task / "card.yaml").read_text())
    card["agent_checks"]["required"][0]["cc"] = ["kev.brannigan@hartleyplumbing.example"]
    (task / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    pkg = load_package(task)
    assert any("required_writes" in w for w in pkg.score(run=pkg.oracle_run)[1])
    _edit_actions(task, lambda d: d["actions"][0].update(cc=["kev.brannigan@hartleyplumbing.example"]))
    pkg = load_package(task)
    assert pkg.score(run=pkg.oracle_run) == (1.0, [])


def test_invented_amount_fails(copy):
    pkg = load_package(AGENT)
    actions = [a.model_copy() for a in pkg.oracle_run.actions]
    actions[0] = actions[0].model_copy(update={"body": actions[0].body + "\nAnd you still owe me £96 from November."})
    _, why = pkg.score(run=OracleRun(actions=actions, tool_calls=7))
    assert any("invented_amounts" in w for w in why)


def test_writing_to_unknown_target_fails():
    pkg = load_package(AGENT)
    extra = Action(tool="create_draft", target="someone@else.example", body="hi")
    _, why = pkg.score(run=OracleRun(actions=[*pkg.oracle_run.actions, extra], tool_calls=8))
    assert any("agent:targets" in w for w in why)


def test_private_phrase_must_live_in_a_private_item(copy):
    task = copy(AGENT)
    card = yaml.safe_load((task / "card.yaml").read_text())
    card["agent_checks"]["private_phrases"] = ["Crestline"]
    (task / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    problems = check_card(load_package(task))
    assert any("not in any private item" in p for p in problems)


def test_similar_prompt_is_rejected(copy, tmp_path):
    task = copy(DIRECT)
    _edit_card(task, id="example-direct-2")
    accepted = tmp_path / "accepted"
    shutil.copytree(DIRECT, accepted / "direct-review")
    problems = check_all(task, accepted_dirs=[accepted])
    assert any("too similar" in p or "same four words" in p for p in problems)


def test_private_leak_in_subject_fails(copy):
    task = copy(AGENT)
    _edit_actions(task, lambda d: d["actions"][0].update(subject="We are £4,820 overdrawn"))
    pkg = load_package(task)
    score, why = pkg.score(run=pkg.oracle_run)
    assert score == 0.0 and any("private_leak" in w for w in why)


def test_empty_required_write_fails(copy):
    task = copy(AGENT)

    def gut(d):
        d["actions"][0]["body"] += "\n\n" + d["actions"][1]["body"]
        d["actions"][1]["body"] = "ok"

    _edit_actions(task, gut)
    pkg = load_package(task)
    assert any("required_writes" in w for w in pkg.score(run=pkg.oracle_run)[1])


def test_recipient_specific_anchor_is_checked(copy):
    task = copy(AGENT)
    card = yaml.safe_load((task / "card.yaml").read_text())
    card["agent_checks"]["required"][1]["must_include"] = [["Wednesday"]]
    card["agent_checks"]["required"][0]["must_include"] = [["collect"]]
    (task / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    pkg = load_package(task)
    assert pkg.score(run=pkg.oracle_run) == (1.0, [])
    _edit_actions(task, lambda d: d["actions"][0].update(body=d["actions"][0]["body"].replace("collect", "pick up")))
    pkg = load_package(task)
    assert any("required_writes" in w for w in pkg.score(run=pkg.oracle_run)[1])


def test_accepted_package_loads_without_oracle(copy):
    task = copy(DIRECT)
    shutil.rmtree(task / "oracle")
    assert load_package(task, require_oracle=False).card.id == "example-direct"
    with pytest.raises(PackageError):
        load_package(task)


def test_rendering_gives_persona_facts_and_length_to_everyone():
    msg = load_package(DIRECT).user_message()
    assert "mobile dog groomer, 28, Halifax, Canada" in msg
    assert "roughly 250-400 words" in msg
    assert "everyday" not in msg  # writing level is how they write, not something they'd say
    assert "all the messages together" in load_package(AGENT).user_message()


def test_authored_block_quotes_count_toward_a_required_write(copy):
    task = copy(AGENT)

    def quote_only(d):
        d["actions"][0]["body"] = "\n".join("> " + w for w in ("filler words that get stripped " * 6).split("  "))
        d["actions"][1]["body"] += "\n\n" + "Priya, the valves are 15mm again and I need 22mm by Wednesday. " * 3

    _edit_actions(task, quote_only)
    pkg = load_package(task)
    assert not any("required_writes" in w for w in pkg.score(run=pkg.oracle_run)[1])


def test_thread_rendering_names_the_user_and_the_message_answered(tmp_path):
    from humanbench.envs import Thread
    pkg = load_package(DIRECT)
    pkg.thread = Thread.model_validate({
        "kind": "forum", "title": "Dryer advice", "user": "halifaxgroomer", "reply_to": "m1",
        "messages": [
            {"id": "m1", "author": "newbie_dan", "date": "2027-01-03", "body": "Which dryer for a van?"},
            {"id": "m2", "author": "someone_else", "date": "2027-01-04", "body": "Following."},
        ],
    })
    msg = pkg.user_message()
    assert "I'm halifaxgroomer here" in msg and "replying to newbie_dan" in msg
    assert "[the message I'm replying to] From: newbie_dan" in msg


def test_staging_dirs_are_not_accepted_packages(tmp_path):
    from humanbench.cardcheck import load_accepted
    root = tmp_path / "dev"
    shutil.copytree(DIRECT, root / "real")
    (root / ".half.staging").mkdir(parents=True)
    (root / ".half.staging" / "card.yaml").write_text((DIRECT / "card.yaml").read_text())
    assert [p.card.id for p in load_accepted([root])] == ["example-direct"]


def test_new_post_rejects_a_reply(copy):
    task = copy(AGENT)
    card = yaml.safe_load((task / "card.yaml").read_text())
    card["agent_checks"]["required"][0] = {"target": "orders@valveco-trade.example", "new_post": True}
    (task / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    pkg = load_package(task)
    assert any("required_writes" in w for w in pkg.score(run=pkg.oracle_run)[1])  # oracle replies to m04
    _edit_actions(task, lambda d: d["actions"][0].update(in_reply_to=None))
    pkg = load_package(task)
    assert pkg.score(run=pkg.oracle_run) == (1.0, [])


def test_banning_the_users_own_word_is_rejected(copy):
    task = copy(DIRECT)
    _edit_card(task, must_not_include=["dryer"])
    assert any("user's own word" in p for p in check_card(load_package(task)))


def test_punctuation_only_anchors_are_rejected(copy):
    task = copy(DIRECT)
    _edit_card(task, must_not_include=["!!"], must_include=[["389"], ["?!"]])
    problems = check_card(load_package(task))
    assert any("would never match" in p for p in problems)
    assert any("empty once punctuation is ignored" in p for p in problems)


def test_derived_amounts_are_not_invented():
    pkg = load_package(AGENT)
    actions = [a.model_copy() for a in pkg.oracle_run.actions]
    body = actions[0].body + "\nThat's £429.60 for two orders of this, and a tenner for the van."  # 2 x 214.80
    actions[0] = actions[0].model_copy(update={"body": body})
    _, why = pkg.score(run=OracleRun(actions=actions, tool_calls=7))
    assert not any("invented_amounts" in w for w in why)


def test_min_words_cannot_exceed_the_band_floor(copy):
    task = copy(AGENT)
    card = yaml.safe_load((task / "card.yaml").read_text())
    card["agent_checks"]["required"][0]["min_words"] = 240
    (task / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    assert any("more than the band's lower bound" in p for p in check_card(load_package(task)))


def test_new_document_may_reference_its_source(copy):
    """new_post fails a comment under a post in the same place, not a new doc that cites its source email."""
    task = copy(AGENT)
    card = yaml.safe_load((task / "card.yaml").read_text())
    card["agent_checks"]["required"][0] = {"target": "orders@valveco-trade.example", "new_post": True}
    (task / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True))
    # cite Kev's email (different sender, not the recipient) as the source of a new email to the supplier
    _edit_actions(task, lambda d: d["actions"][0].update(in_reply_to="m08"))
    pkg = load_package(task)
    assert pkg.score(run=pkg.oracle_run) == (1.0, [])


def test_card_length_override_must_touch_the_band():
    import pytest
    from humanbench.schema import TaskCard
    import yaml
    from pathlib import Path

    raw = yaml.safe_load((Path(__file__).parents[1] / "tasks" / "dev" / "social-08" / "card.yaml").read_text())
    card = TaskCard.model_validate(raw)
    assert card.words == (360, 400) and card.max_chars == 3000
    with pytest.raises(ValueError):
        TaskCard.model_validate({**raw, "length_words": [1500, 1900]})
