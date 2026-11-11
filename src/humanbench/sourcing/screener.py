"""Recruiting screeners: a Persona turned into machine-readable criteria.

The output is data for a Prolific-style screener (or Upwork / university
lists for niche roles). Nothing here calls a recruiting API. Field names are
generic; map them to the platform's own filter ids when you set up a study.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from humanbench.axes import Channel, WritingLevel
from humanbench.schema import Persona

PLATFORM_MIN_AGE = 18  # Prolific and most panels are adults only

# English variety -> where the writer grew up (map to nationality / country of birth).
VARIETY_COUNTRY: dict[str, list[str]] = {
    "US": ["United States"],
    "UK": ["United Kingdom"],
    "Scottish": ["United Kingdom"],
    "Welsh": ["United Kingdom"],
    "Irish": ["Ireland"],
    "Australian": ["Australia"],
    "Canadian": ["Canada"],
    "New Zealand": ["New Zealand"],
    "South African": ["South Africa"],
    "Indian": ["India"],
    "Nigerian": ["Nigeria"],
    "Ghanaian": ["Ghana"],
    "Kenyan": ["Kenya"],
    "Singaporean": ["Singapore"],
    "Jamaican": ["Jamaica"],
    "Filipino": ["Philippines"],
}
VARIETY_REGION = {"Scottish": "Scotland", "Welsh": "Wales"}
# Varieties whose speakers usually have English as their first language. In the
# others, many fluent speakers grew up with another first language too, so a
# "first language: English" filter would wrongly exclude them.
ENGLISH_L1_VARIETIES = {"US", "UK", "Scottish", "Welsh", "Irish", "Australian", "Canadian", "New Zealand"}

_US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
}
_UK_PARTS = {"England", "Scotland", "Wales", "Northern Ireland", "London"}

# First match wins, so more specific sectors come first. Keys match at word starts.
_SECTOR_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Retired", ("retired",)),
    ("Student", ("student", "undergraduate", "phd", "high-school junior", "fresher")),
    ("Veterinary", ("veterinary",)),
    ("Healthcare", ("nurse", "doctor", "midwife", "pharmacist", "physio", "medicine", "hospice", "epidemiolog")),
    ("Education", ("teacher", "coach", "librarian", "tutor", "lecturer")),
    ("Research / academia", ("postdoc", "researcher", "biologist", "scientist")),
    ("Media / creative", ("journalist", "photographer", "tattoo", "musician", "writer", "game developer")),
    ("Software / IT", ("software engineer", "developer", "sre", "site reliability", "maintainer", "product designer",
                       "it-support")),
    ("Engineering", ("engineer",)),
    ("Legal", ("lawyer", "paralegal", "attorney", "solicitor")),
    ("Hospitality / food", ("chef", "sous-chef", "barista", "bartender", "bakery", "restaurant", "cafe")),
    ("Personal services", ("hairdresser", "groomer", "nail-salon")),
    ("Logistics", ("warehouse",)),
    ("Skilled trades", ("plumber", "electrician", "mechanic", "locksmith", "gardener", "carpenter", "builder")),
    ("Agriculture", ("farmer", "dairy", "ranch")),
    ("Sales / business", ("sales", "account executive", "founder", "real-estate", "realtor", "owner", "accountant", "hr",
                          "event planner", "trader", "reseller")),
    ("Transport", ("truck", "driver", "flight attendant", "pilot")),
    ("Public sector", ("council", "government", "department")),
    ("Religion", ("pastor", "priest", "imam", "rabbi")),
]

WRITING_SCREEN: dict[WritingLevel, str] = {
    WritingLevel.POLISHED: "Writes regularly for work or publication; sample is edited, varied and nearly error-free.",
    WritingLevel.EVERYDAY: "Writes emails and messages comfortably; sample is clear with ordinary informality.",
    WritingLevel.ROUGH: "Writes mostly informally; sample has spelling or grammar slips, but the meaning is always clear.",
}


class Criterion(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    op: Literal["between", "in", "equals", "includes", "self_report", "rated_sample"]
    value: Any
    required: bool = True
    note: str = ""


class ScreenerSpec(BaseModel):
    persona_id: str
    title: str
    age_min: int
    age_max: int
    involves_minor: bool
    english_variety: str | None
    first_language: list[str]  # empty means "don't filter on first language"
    fluent_languages: list[str]
    grew_up_in: list[str]
    current_country: list[str]
    current_location: str
    occupation: str
    occupation_sector: str | None
    writing_level: WritingLevel
    writing_screen: str
    excluded_channels: list[Channel]
    criteria: list[Criterion]
    notes: list[str]


def age_window(age: int) -> tuple[int, int]:
    tol = 2 if age < 22 else 4 if age < 35 else 6 if age < 60 else 8
    return max(age - tol, 13), age + tol


def country_of(lives: str) -> str | None:
    last = lives.split(",")[-1].strip()
    if last in _US_STATES:
        return "United States"
    if last in _UK_PARTS:
        return "United Kingdom"
    return last or None


def sector_of(role: str) -> str | None:
    r = role.lower()
    for sector, keys in _SECTOR_KEYWORDS:
        if any(re.search(rf"\b{re.escape(k)}\b" if len(k) <= 3 else rf"\b{re.escape(k)}", r) for k in keys):
            return sector
    return None


def screener_for(persona: Persona) -> ScreenerSpec:
    lo, hi = age_window(persona.age)
    minor = persona.age < PLATFORM_MIN_AGE
    notes: list[str] = []
    criteria: list[Criterion] = []

    if minor:
        notes.append(
            f"Persona is {persona.age}. Panels like Prolific are 18+, so recruit {PLATFORM_MIN_AGE}-{hi} there, "
            "or recruit under-18s only through a channel with guardian consent and ethics approval."
        )
    # Panels are adults only, so the window never starts below 18 (an 18-year-old persona gets 18-20).
    criteria.append(Criterion(field="age", op="between", value=[max(lo, PLATFORM_MIN_AGE), hi],
                              note="widen the window before compromising on the role"))

    if persona.native:
        variety = persona.english
        grew_up = VARIETY_COUNTRY.get(variety, [])
        first_language = ["English"] if variety in ENGLISH_L1_VARIETIES else []
        if not first_language:
            notes.append(f"{variety} English: don't require English as first language; many writers are multilingual.")
        if region := VARIETY_REGION.get(variety):
            notes.append(f"Grew up in {region} specifically (the platform filter is usually only the UK).")
            criteria.append(Criterion(field="grew_up_region", op="self_report", value=region))
        if grew_up:
            criteria.append(Criterion(field="grew_up_in", op="in", value=grew_up,
                                      note="map to nationality or country of birth"))
        if first_language:
            criteria.append(Criterion(field="first_language", op="in", value=first_language))
    else:
        variety = None
        grew_up = []
        first_language = [persona.first_language or ""]
        criteria.append(Criterion(field="first_language", op="in", value=first_language))
        notes.append("Non-native writer: the reference must be their own English, unedited by anyone else.")
    criteria.append(Criterion(field="fluent_languages", op="includes", value="English"))

    current = country_of(persona.lives)
    current_country = [current] if current else []
    if current_country:
        criteria.append(Criterion(field="current_country", op="in", value=current_country, required=False,
                                  note="preferred; the role and English variety matter more"))

    sector = sector_of(persona.role)
    if sector:
        criteria.append(Criterion(field="employment_sector", op="equals", value=sector, required=False))
    criteria.append(Criterion(field="occupation", op="self_report", value=persona.role,
                              note="free-text answer, checked by a person; must match the role closely"))
    criteria.append(Criterion(field="writing_sample", op="rated_sample", value=str(persona.writing_level),
                              note=WRITING_SCREEN[persona.writing_level]))
    notes.append("Never tell writers which writing level they were matched to; it is a pool-balancing label.")

    return ScreenerSpec(
        persona_id=persona.id,
        title=re.sub(r"\s+", " ", f"{persona.role}, about {persona.age}"),
        age_min=lo,
        age_max=hi,
        involves_minor=minor,
        english_variety=variety,
        first_language=first_language,
        fluent_languages=["English"],
        grew_up_in=grew_up,
        current_country=current_country,
        current_location=persona.lives,
        occupation=persona.role,
        occupation_sector=sector,
        writing_level=persona.writing_level,
        writing_screen=WRITING_SCREEN[persona.writing_level],
        excluded_channels=list(persona.excluded_channels),
        criteria=criteria,
        notes=notes,
    )
