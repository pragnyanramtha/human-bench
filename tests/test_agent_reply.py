from pathlib import Path

import pytest

from humanbench.taskpkg import load_package

PKG = load_package(Path(__file__).parent / "fixtures" / "agent-email", require_oracle=False)
T = PKG.env.targets[0]


def test_plain_draft_with_header_is_parsed():
    a = PKG.parse_agent_reply(f"To: {T}\nReplying to: none\nCc: none\nSubject: About Thursday\n\nHi Sam,\n\nSee you then.\n\nJo")
    assert len(a) == 1 and a[0].target == T and a[0].in_reply_to is None and a[0].cc == []
    assert a[0].subject == "About Thursday" and a[0].body == "Hi Sam,\n\nSee you then.\n\nJo"


def test_two_messages_markdown_headers_and_reply_ids():
    text = (f"**To:** {T}\n**Replying to:** [m02]\n**Cc:** a@x.example, b@x.example\n**Subject:** none\n\nFirst body.\n"
            f"\n---\n\nTo: {T}\nReplying to: None\nCc: none\nSubject: none\n\nSecond body.")
    a = PKG.parse_agent_reply(text)
    assert [x.body for x in a] == ["First body.", "Second body."]
    assert a[0].in_reply_to == "m02" and a[0].cc == ["a@x.example", "b@x.example"] and a[0].subject is None


def test_a_dashed_line_inside_a_message_stays_in_that_message():
    a = PKG.parse_agent_reply(f"To: {T}\nReplying to: none\nCc: none\nSubject: Hi\n\nPart one.\n\n---\n\nPart two, same message.")
    assert len(a) == 1 and "Part two" in a[0].body


def test_old_json_shape_still_accepted_and_garbage_is_rejected():
    a = PKG.parse_agent_reply('{"actions": [{"tool": "create_draft", "target": "' + T + '", "body": "Hello"}]}')
    assert a[0].body == "Hello"
    with pytest.raises(ValueError):
        PKG.parse_agent_reply("Sure! Here is the email you asked for. Dear Sam, see you Thursday.")


def test_the_prompt_ends_with_the_draft_format_and_the_length():
    p = PKG.model_prompt()
    assert "Reply with only JSON" not in p and "To: <where this text is to end up" in p
    first = PKG.env.items[0].id  # the instructions must not offer a real item id as their example
    assert f"such as {first}" not in p and "Replying to: <only if this text is itself a reply" in p
    assert p.rstrip().split("\n")[-1].startswith("Length: all the messages together")


def test_several_recipients_on_the_to_line_are_one_message_to_all_of_them():
    a = PKG.parse_agent_reply("To: Jan <jan@x.example>, pete@x.example; carol@x.example\nReplying to: [m01]\n"
                              "Cc: dana@x.example and eli@x.example\nSubject: none\n\nHello all.")
    assert len(a) == 1 and a[0].target == "jan@x.example"
    assert a[0].cc == ["pete@x.example", "carol@x.example", "dana@x.example", "eli@x.example"]
    assert a[0].in_reply_to == "m01"


def test_a_valid_target_is_taken_whole():
    a = PKG.parse_agent_reply(f"To: {T}\nReplying to: none\nCc: none\nSubject: none\n\nHello.")
    assert a[0].target == T and a[0].cc == []


def test_reply_values_with_words_round_the_id_or_no_id_at_all():
    first = PKG.env.items[0].id

    def reply(value: str) -> str | None:
        return PKG.parse_agent_reply(f"To: {T}\nReplying to: {value}\nCc: none\nSubject: none\n\nHello.")[0].in_reply_to

    for v in ("none", "None - this is a new email", "none (new message)", "[ ]", "( )", "—", "-", "n/a", ""):
        assert reply(v) is None, v
    for v in (first, f"[{first}]", f"{first}, and also the one after", f"the message {first} from Monday"):
        assert reply(v) == first, v
    assert reply("ZZ-999") == "ZZ-999"  # an id the workspace does not have stays visible to the check


def test_names_brackets_and_notes_round_a_target_do_not_hide_it():
    a = PKG.parse_agent_reply(f"To: Okafor, Sam <{T}> (draft)\nReplying to: none\nCc: none needed\nSubject: none\n\nHello.")
    assert a[0].target == T and a[0].cc == []
    a = PKG.parse_agent_reply(f"To: someone.else@nowhere.example, {T}\nReplying to: none\nCc: none\nSubject: none\n\nHello.")
    assert a[0].target == "someone.else@nowhere.example" and a[0].cc == [T]  # an address the card does not list is kept


def test_two_messages_with_only_a_blank_line_between_them_are_two_messages():
    text = (f"To: {T}\nReplying to: none\nCc: none\nSubject: First\n\nFirst message, with a line that says\n"
            "To: do the thing, you need a list.\n\n"
            f"To: {T}\nReplying to: none\nCc: none\nSubject: Second\n\nSecond message.")
    a = PKG.parse_agent_reply(text)
    assert [x.subject for x in a] == ["First", "Second"]
    assert "you need a list" in a[0].body and "Second message." == a[1].body
