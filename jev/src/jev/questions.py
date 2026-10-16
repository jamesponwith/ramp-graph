"""Question set v0, by hand: what an entity's text says that its counts do not.

Each is about the entity alone, so it is asked once per entity, not per task. Relevance
to the analyst's prompt would be asked per (task, candidate); it waits for P3.
"""

import math

import numpy as np

QUESTIONS: dict[str, dict] = {
    "nominee": {
        "type": "noul",
        "instructions": "Is this a nominee director, nominee shareholder, corporate "
        "secretary or registered agent acting for others, rather than a party acting "
        "for itself?",
    },
    "provider": {
        "type": "noul",
        "instructions": "Is this a law firm, bank, trust company or corporate service "
        "provider that sets up or administers companies for clients?",
    },
    "mass_address": {
        "type": "noul",
        "instructions": "Is this, or does this entity use, a registered-office or "
        "mass-registration address shared by many unrelated companies?",
    },
    "hub": {
        "type": "score",
        "instructions": "How many unrelated parties is this entity likely linked to in "
        "an offshore-leaks database?",
        "criteria": ["one or two", "a handful", "dozens", "hundreds or more"],
    },
    "person": {"type": "noul", "instructions": "Is this a natural person?"},
    "specific": {
        "type": "noul",
        "instructions": "Does the text name one specific, identifiable party, rather "
        'than a placeholder or generic holder such as "bearer", "the shareholders" or '
        "an unnamed nominee?",
    },
    "secrecy": {
        "type": "noul",
        "instructions": "Is the jurisdiction, or the country given, a secrecy "
        "jurisdiction or tax haven?",
    },
    "shell": {
        "type": "noul",
        "instructions": "Is this likely a shell company with no operating business of its own?",
    },
    "active": {
        "type": "noul",
        "instructions": "Is this entity still active: not struck off, dissolved, "
        "defaulted or inactive?",
    },
    "notable": {
        "type": "noul",
        "instructions": "Is this a well-known company, public figure or politically "
        "exposed person?",
    },
}

NAMES = [f"jev_{q}" for q in QUESTIONS]


def features(
    ids: np.ndarray, answers: dict[int, dict[str, dict | None]], qs: dict = QUESTIONS
) -> np.ndarray:
    """One row per id, a column per question in `qs`."""
    uniq, inv = np.unique(ids, return_inverse=True)
    table = np.array([[value(answers[i][q]) for q in qs] for i in uniq], np.float32)
    return table[inv]


def value(answer: dict | None) -> float:
    """A noul's probability, a score's expected level scaled to [0, 1], or NaN for no
    answer."""
    if answer is None:
        return math.nan
    if answer["type"] == "noul":
        return answer["noul"]
    return answer["score"] / max(len(answer["legend"]) - 1, 1)
