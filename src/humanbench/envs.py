"""Fixtures for thread-reply, agent and ghostwrite tasks (DESIGN.md 2.2, 4.3).

A task directory holds at most three fixture files next to `card.yaml`:

    context.json         Thread      thread_reply: the thread the user answers
    env/fixture.json     Environment agent/ghostwrite: inbox, Slack, forum, shop...
    env/user_voice.json  UserVoice   ghostwrite: the user's own past writing

The harness exposes one small tool library over an Environment (`TOOL_LIBRARY`).
Read tools return items; write tools record an `Action`. Agent checks run on the
recorded actions, never on the transcript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Who wrote a fixture's text. Model-drafted thread messages and voice samples are
# placeholders: a model imitating model-written "user voice" would be rewarded for
# sounding like a model, so they are replaced by human writing before freeze.
Provenance = Literal["model_draft", "human"]

READ_TOOLS = ("list_items", "search", "read_item", "open_attachment")
WRITE_TOOLS = ("create_draft", "send")
TOOL_LIBRARY = READ_TOOLS + WRITE_TOOLS


class ThreadMessage(BaseModel):
    id: str
    author: str
    date: str
    body: str = Field(min_length=1)
    subject: str | None = None
    to: list[str] = []
    reply_to: str | None = None


class Thread(BaseModel):
    """The thread a thread_reply task answers."""

    kind: Literal["email", "chat", "forum"]
    title: str | None = None  # email subject, Slack channel or DM name, forum thread title
    where: str | None = None  # mailbox, workspace or board
    user: str  # the persona's name or handle in this thread
    messages: list[ThreadMessage] = Field(min_length=2)
    reply_to: str  # id of the message the user is answering
    provenance: Provenance = "model_draft"

    @model_validator(mode="after")
    def _references(self) -> Thread:
        ids = [m.id for m in self.messages]
        if len(set(ids)) != len(ids):
            raise ValueError("thread message ids must be unique")
        if self.reply_to not in ids:
            raise ValueError(f"reply_to {self.reply_to!r} is not a message id")
        for m in self.messages:
            if m.reply_to is not None and m.reply_to not in ids:
                raise ValueError(f"message {m.id}: reply_to {m.reply_to!r} is not a message id")
        return self


class Attachment(BaseModel):
    name: str
    content: str


ItemKind = Literal["email", "message", "post", "order", "file", "event", "note", "review", "listing"]


class Item(BaseModel):
    id: str
    kind: ItemKind
    folder: str  # inbox, sent, #channel, @dm, board, orders, docs, calendar, feed
    author: str | None = None
    to: list[str] = []
    date: str
    title: str | None = None
    body: str = ""
    thread: str | None = None
    reply_to: str | None = None
    attachments: list[Attachment] = []
    private: bool = False  # must never leak into anything the agent writes


class Environment(BaseModel):
    """A fictional workspace an agent task runs in."""

    kind: Literal["inbox", "slack", "forum", "social", "shop", "cms", "docs"]
    owner: str  # the user's name, address or handle
    now: str  # the current date and time inside the world
    items: list[Item] = Field(min_length=6)
    targets: list[str] = Field(min_length=1)  # where create_draft / send may write
    provenance: Provenance = "model_draft"

    @model_validator(mode="after")
    def _references(self) -> Environment:
        ids = [i.id for i in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("environment item ids must be unique")
        for item in self.items:
            if item.reply_to is not None and item.reply_to not in ids:
                raise ValueError(f"item {item.id}: reply_to {item.reply_to!r} is not an item id")
        return self

    def item(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)

    def texts(self) -> list[str]:
        out: list[str] = []
        for i in self.items:
            out += [i.title or "", i.body] + [a.content for a in i.attachments]
        return [t for t in out if t]


class VoiceSample(BaseModel):
    id: str
    date: str
    where: str
    title: str | None = None
    body: str = Field(min_length=1)


class UserVoice(BaseModel):
    """Past writing by the user, for ghostwrite tasks."""

    samples: list[VoiceSample] = Field(min_length=3)
    provenance: Provenance = "model_draft"


class Action(BaseModel):
    """One write the agent made. The harness records these from tool calls."""

    tool: Literal["create_draft", "send"]
    target: str
    body: str
    subject: str | None = None
    in_reply_to: str | None = None
    cc: list[str] = []  # email only


class ExpectedWrite(BaseModel):
    target: str | list[str]  # any of these targets counts
    in_reply_to: str | None = None  # this exact item, or (None) any
    new_post: bool = False  # must start something new: no in_reply_to at all
    all_recipients: list[str] = []  # one write that reaches all of these, in To or Cc in any order
    cc: list[str] = []  # these must all be copied on the matching write
    min_words: int = Field(default=25, ge=1)  # the matching write must say something
    must_include: list[str | list[str]] = []  # recipient-specific anchors, in this write's body

    def targets(self) -> list[str]:
        return [self.target] if isinstance(self.target, str) else list(self.target)

    @model_validator(mode="after")
    def _reply_policy(self) -> ExpectedWrite:
        if self.new_post and self.in_reply_to is not None:
            raise ValueError("new_post writes cannot also name in_reply_to")
        return self


class AgentChecks(BaseModel):
    """What a correct agent run must do (DESIGN.md 4.3)."""

    delivery: Literal["draft", "send"]
    required: list[ExpectedWrite] = Field(min_length=1)
    forbidden_targets: list[str] = []
    private_phrases: list[str] = []  # distinctive strings from private items
    max_tool_calls: int = Field(default=20, ge=1, le=20)
    max_writes: int | None = None  # default: len(required) + 1


class OracleRun(BaseModel):
    """The golden agent run kept in oracle/actions.json."""

    actions: list[Action] = Field(min_length=1)
    tool_calls: int = Field(ge=1)


def load_json_model[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
