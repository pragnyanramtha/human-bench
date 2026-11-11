"""Data models for personas, task slots and full task cards."""

from __future__ import annotations

from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from humanbench.axes import (
    ALLOWED_AUDIENCES,
    ALLOWED_BANDS,
    BAND_WORDS,
    Audience,
    Band,
    Channel,
    Mode,
    Stakes,
    Tone,
    WritingLevel,
)
from humanbench.envs import TOOL_LIBRARY, AgentChecks

Split = Literal["dev", "test", "ref"]  # ref: built backwards from a real human text


class Persona(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    role: str
    age: int | None = Field(default=None, ge=13, le=100)  # None only on reference cards, when the text gives no age
    english: str  # variety of English, or "L2" for non-native writers
    first_language: str | None = None
    lives: str | None = None  # None only on reference cards, when the text does not say
    writing_level: WritingLevel
    excluded_channels: tuple[Channel, ...] = ()

    @property
    def native(self) -> bool:
        return self.english != "L2"

    @property
    def non_us(self) -> bool:
        return self.english != "US"

    @model_validator(mode="after")
    def _l2_needs_first_language(self) -> Persona:
        if (self.english == "L2") != (self.first_language is not None):
            raise ValueError(f"{self.id}: first_language must be set exactly when english is L2")
        return self


def load_personas() -> list[Persona]:
    raw = resources.files("humanbench.data").joinpath("personas.yaml").read_text(encoding="utf-8")
    personas = [Persona.model_validate(p) for p in yaml.safe_load(raw)]
    ids = [p.id for p in personas]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate persona ids in personas.yaml")
    return personas


class Slot(BaseModel):
    """One cell of the task matrix: everything about a task except its content."""

    id: str
    split: Split
    channel: Channel
    mode: Mode
    tone: Tone
    audience: Audience
    stakes: Stakes
    band: Band
    persona_id: str

    @property
    def words(self) -> tuple[int, int]:
        return BAND_WORDS[self.band]

    @model_validator(mode="after")
    def _combination_allowed(self) -> Slot:
        if (self.channel, self.mode) not in ALLOWED_BANDS:
            raise ValueError(f"{self.id}: mode {self.mode} is not used in channel {self.channel}")
        if self.band not in ALLOWED_BANDS[(self.channel, self.mode)]:
            raise ValueError(f"{self.id}: band {self.band} not allowed for {self.channel}/{self.mode}")
        if self.audience not in ALLOWED_AUDIENCES[self.channel]:
            raise ValueError(f"{self.id}: audience {self.audience} not allowed for {self.channel}")
        return self


class NotesPack(BaseModel):
    """Raw material shared by the human writer and the model."""

    facts: list[str] = []
    anecdotes: list[str] = []
    opinions: list[str] = []
    wants: list[str] = []


# A must_include entry is a phrase, or a list of alternative phrases (any one counts).
Requirement = str | list[str]


class TaskCard(Slot):
    """A finished task: a slot plus the prompt, notes and environment."""

    persona: Persona
    prompt: str = Field(min_length=20)
    notes_pack: NotesPack = NotesPack()
    context: str | None = None  # thread file for thread_reply
    environment: str | None = None  # fixture directory for agent/ghostwrite
    tools: list[str] = []
    user_voice: str | None = None  # past writing of the user, for ghostwrite
    must_include: list[Requirement] = []
    must_not_include: list[str] = []
    # What the piece has to get across, in any words: checked by the quality floor, never by string
    # matching. Each point is something the request explicitly asks for. Hidden from models and writers.
    must_cover: list[str] = Field(default=[], max_length=8)
    # What the user explicitly asks to keep out, as plain statements ("that Raff's father was in hospital").
    # The quality floor checks them by meaning, like items marked private in a workspace.
    must_not_reveal: list[str] = Field(default=[], max_length=5)
    # When the user states their own length (a platform limit, "keep it to 360-400 words"), the card says so
    # here, so that what the model is told and what is checked agree. Must touch the band's range.
    length_words: tuple[int, int] | None = None
    max_chars: int | None = Field(default=None, ge=500)
    format_overrides: dict[str, bool] = {}
    agent_checks: AgentChecks | None = None  # agent and ghostwrite tasks
    human_refs: list[str] = []
    drafted_by: str | None = None  # model or person who drafted the card

    @property
    def words(self) -> tuple[int, int]:
        return self.length_words or BAND_WORDS[self.band]

    @model_validator(mode="after")
    def _length_inside_band(self) -> TaskCard:
        if self.length_words is not None:
            lo, hi = self.length_words
            blo, bhi = BAND_WORDS[self.band]
            # The user's own range may sit at the edge of the band, but must touch it and stay in 250-4000.
            if not (250 <= lo < hi <= 4000 and hi >= blo * 0.9 and lo <= bhi * 1.1):
                raise ValueError(f"{self.id}: length_words {self.length_words} does not fit the band {blo}-{bhi}")
        return self

    @model_validator(mode="after")
    def _mode_requirements(self) -> TaskCard:
        if self.persona.id != self.persona_id:
            raise ValueError(f"{self.id}: persona does not match persona_id")
        if self.mode is Mode.THREAD and not self.context:
            raise ValueError(f"{self.id}: thread_reply tasks need a context thread")
        if self.mode in (Mode.AGENT, Mode.GHOSTWRITE):
            if not self.environment or not self.tools:
                raise ValueError(f"{self.id}: {self.mode} tasks need an environment and tools")
            if self.agent_checks is None:
                raise ValueError(f"{self.id}: {self.mode} tasks need agent_checks")
            unknown = sorted(set(self.tools) - set(TOOL_LIBRARY))
            if unknown:
                raise ValueError(f"{self.id}: unknown tools {unknown}; allowed: {list(TOOL_LIBRARY)}")
        elif self.agent_checks is not None or self.tools:
            raise ValueError(f"{self.id}: only agent and ghostwrite tasks take tools and agent_checks")
        if self.mode is Mode.GHOSTWRITE and not self.user_voice:
            raise ValueError(f"{self.id}: ghostwrite tasks need user_voice samples")
        return self
