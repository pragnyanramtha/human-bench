"""Benchmark axes and v1 allocation targets.

Single source of truth for the numbers in DESIGN.md section 2. If you change a
target here, update DESIGN.md too.
"""

from enum import StrEnum


class Channel(StrEnum):
    EMAIL = "email"
    CHAT = "chat"
    FORUM = "forum"
    SOCIAL = "social"
    REVIEW = "review"
    BLOG = "blog"
    WORK_DOCS = "work_docs"
    PERSONAL = "personal"


class Mode(StrEnum):
    DIRECT = "direct"
    THREAD = "thread_reply"
    AGENT = "agent"
    GHOSTWRITE = "ghostwrite"


class Tone(StrEnum):
    CASUAL = "casual"
    EMOTIONAL = "emotional"
    ANGRY = "angry"
    FUNNY = "funny"
    EXCITED = "excited"
    PERSUASIVE = "persuasive"
    FORMAL = "formal"
    TECHNICAL = "technical"


class Audience(StrEnum):
    FRIEND = "friend"
    FAMILY = "family"
    BOSS = "boss"
    COLLEAGUE = "colleague"
    STRANGER = "stranger"
    CUSTOMER = "customer"
    COMMUNITY = "community"
    PUBLIC = "public"
    BUSINESS = "business"  # a company or organisation being addressed


class Stakes(StrEnum):
    ROUTINE = "routine"
    TENSE = "tense"
    CRISIS = "crisis"


class WritingLevel(StrEnum):
    POLISHED = "polished"
    EVERYDAY = "everyday"
    ROUGH = "rough"  # rough but clear


class Band(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    EXTENDED = "extended"


BAND_WORDS: dict[Band, tuple[int, int]] = {
    Band.SHORT: (250, 400),
    Band.MEDIUM: (400, 800),
    Band.LONG: (800, 2000),
    Band.EXTENDED: (2000, 4000),
}

BAND_TARGETS: dict[Band, int] = {
    Band.SHORT: 64,
    Band.MEDIUM: 56,
    Band.LONG: 28,
    Band.EXTENDED: 12,
}

TASKS_PER_CHANNEL = 20
DEV_PER_CHANNEL = 10  # 80 dev / 80 test (decided 2026-11-09; was 40 / 120)
MAX_PERSONA_USES = 3
MIN_TONES_PER_CHANNEL = 5
MIN_AUDIENCES_PER_CHANNEL = 5
MIN_NON_US_SHARE = 0.30
NON_NATIVE_SHARE_RANGE = (0.10, 0.25)
MIN_AGE_SPAN_PER_CHANNEL = 30
STAKES_PER_CHANNEL: dict[Stakes, int] = {Stakes.ROUTINE: 9, Stakes.TENSE: 7, Stakes.CRISIS: 4}

C, M = Channel, Mode
CHANNEL_MODE_COUNTS: dict[Channel, dict[Mode, int]] = {
    C.EMAIL: {M.DIRECT: 4, M.THREAD: 6, M.AGENT: 6, M.GHOSTWRITE: 4},
    C.CHAT: {M.DIRECT: 2, M.THREAD: 10, M.AGENT: 8},
    C.FORUM: {M.DIRECT: 2, M.THREAD: 12, M.AGENT: 6},
    C.SOCIAL: {M.DIRECT: 8, M.AGENT: 6, M.GHOSTWRITE: 6},
    C.REVIEW: {M.DIRECT: 12, M.AGENT: 8},
    C.BLOG: {M.DIRECT: 14, M.GHOSTWRITE: 6},
    C.WORK_DOCS: {M.DIRECT: 12, M.THREAD: 2, M.AGENT: 6},
    C.PERSONAL: {M.DIRECT: 20},
}

B = Band
_SM = frozenset({B.SHORT, B.MEDIUM})
ALLOWED_BANDS: dict[tuple[Channel, Mode], frozenset[Band]] = {
    (C.EMAIL, M.DIRECT): _SM,
    (C.EMAIL, M.THREAD): _SM,
    (C.EMAIL, M.AGENT): _SM,
    (C.EMAIL, M.GHOSTWRITE): _SM,
    (C.CHAT, M.DIRECT): _SM,
    (C.CHAT, M.THREAD): _SM,
    (C.CHAT, M.AGENT): _SM,
    (C.FORUM, M.DIRECT): frozenset({B.SHORT, B.MEDIUM, B.LONG}),
    (C.FORUM, M.THREAD): frozenset({B.SHORT, B.MEDIUM, B.LONG}),
    (C.FORUM, M.AGENT): _SM,
    (C.SOCIAL, M.DIRECT): _SM,
    (C.SOCIAL, M.AGENT): _SM,
    (C.SOCIAL, M.GHOSTWRITE): _SM,
    (C.REVIEW, M.DIRECT): _SM,
    (C.REVIEW, M.AGENT): _SM,
    (C.BLOG, M.DIRECT): frozenset({B.MEDIUM, B.LONG, B.EXTENDED}),
    (C.BLOG, M.GHOSTWRITE): frozenset({B.MEDIUM, B.LONG}),
    (C.WORK_DOCS, M.DIRECT): frozenset({B.SHORT, B.MEDIUM, B.LONG, B.EXTENDED}),
    (C.WORK_DOCS, M.THREAD): _SM,
    (C.WORK_DOCS, M.AGENT): frozenset({B.MEDIUM, B.LONG}),
    (C.PERSONAL, M.DIRECT): frozenset({B.SHORT, B.MEDIUM, B.LONG, B.EXTENDED}),
}

A = Audience
ALLOWED_AUDIENCES: dict[Channel, tuple[Audience, ...]] = {
    C.EMAIL: (A.FRIEND, A.FAMILY, A.BOSS, A.COLLEAGUE, A.STRANGER, A.CUSTOMER, A.BUSINESS),
    C.CHAT: (A.FRIEND, A.FAMILY, A.BOSS, A.COLLEAGUE, A.COMMUNITY),
    C.FORUM: (A.STRANGER, A.COMMUNITY, A.PUBLIC),
    C.SOCIAL: (A.FRIEND, A.FAMILY, A.COLLEAGUE, A.CUSTOMER, A.COMMUNITY, A.PUBLIC),
    C.REVIEW: (A.PUBLIC, A.COMMUNITY, A.BUSINESS),
    C.BLOG: (A.FRIEND, A.FAMILY, A.CUSTOMER, A.COMMUNITY, A.PUBLIC),
    C.WORK_DOCS: (A.BOSS, A.COLLEAGUE, A.CUSTOMER, A.COMMUNITY, A.PUBLIC),
    C.PERSONAL: (A.FRIEND, A.FAMILY, A.STRANGER, A.COMMUNITY, A.PUBLIC),
}


def total_tasks() -> int:
    return sum(sum(modes.values()) for modes in CHANNEL_MODE_COUNTS.values())


def min_audiences(channel: Channel) -> int:
    return min(MIN_AUDIENCES_PER_CHANNEL, len(ALLOWED_AUDIENCES[channel]))
