"""Turn extracted clauses into User/Assistant pairs.

The assistant line of a question or a "say a sentence" pair is the source
sentence. Slots only choose the question, and each slot string must occur
in that sentence.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Clause:
    sentence: str
    subject: str | None
    subject_proper: bool
    verb: str
    verb_lemma: str
    verb_tag: str
    obj: str | None
    prep: str | None
    pobj: str | None
    pobj_time: bool
    pobj_place: bool
    noun: str


@dataclass(frozen=True)
class Pair:
    user: str
    assistant: str


_IRREGULAR_3SG = {"be": "is", "have": "has", "do": "does", "go": "goes"}
_FINITE = {"VB", "VBP", "VBZ", "VBD"}


def third_singular(lemma: str) -> str:
    key = lemma.casefold()
    if key in _IRREGULAR_3SG:
        return _IRREGULAR_3SG[key]
    if key.endswith(("s", "x", "z", "ch", "sh")):
        return lemma + "es"
    return lemma + "s"


def _has(sentence: str, part: str) -> bool:
    return part.casefold() in sentence.casefold()


def _grounded(sentence: str, parts: list[str]) -> bool:
    return all(_has(sentence, part) for part in parts if part)


def _do_support(tag: str) -> str | None:
    if tag == "VBZ":
        return "does"
    if tag in {"VBP", "VB"}:
        return "do"
    if tag == "VBD":
        return "did"
    return None


def _pair(sentence: str, user: str, assistant: str, slots: list[str]) -> Pair | None:
    user = " ".join(user.split())
    assistant = " ".join(assistant.split())
    if not user or not assistant or "\n" in user or "\n" in assistant:
        return None
    if not _grounded(sentence, slots):
        return None
    return Pair(user, assistant)


def _subject_verb(clause: Clause) -> tuple[str, str] | None:
    if clause.verb_tag not in _FINITE or clause.verb_tag == "VB":
        return None
    if clause.verb_tag == "VBP":
        return third_singular(clause.verb_lemma), clause.verb_lemma
    return clause.verb, clause.verb


def _tail(clause: Clause, verb: str, ground_verb: str) -> tuple[str, list[str]] | None:
    words = [verb]
    slots = [ground_verb]
    if clause.obj:
        words.append(clause.obj)
        slots.append(clause.obj)
    if clause.prep and clause.pobj:
        words.append(clause.prep)
        words.append(clause.pobj)
        slots.append(clause.prep)
        slots.append(clause.pobj)
    if not _grounded(clause.sentence, slots):
        return None
    return " ".join(words), slots


def _subject_question(clause: Clause) -> Pair | None:
    if not clause.subject:
        return None
    picked = _subject_verb(clause)
    if picked is None:
        return None
    verb, ground_verb = picked
    tail = _tail(clause, verb, ground_verb)
    if tail is None:
        return None
    words, slots = tail
    wh = "Who" if clause.subject_proper else "What"
    return _pair(clause.sentence, f"{wh} {words}?", clause.sentence, slots)


def _object_question(clause: Clause) -> Pair | None:
    if not clause.subject or not clause.obj or clause.verb_lemma.casefold() == "be":
        return None
    aux = _do_support(clause.verb_tag)
    if aux is None:
        return None
    slots = [clause.subject, clause.verb_lemma, clause.obj]
    return _pair(
        clause.sentence,
        f"What {aux} {clause.subject} {clause.verb_lemma}?",
        clause.sentence,
        slots,
    )


def _prep_question(clause: Clause) -> Pair | None:
    if not clause.subject or not clause.prep or not clause.pobj:
        return None
    if clause.verb_lemma.casefold() == "be":
        return None
    if clause.pobj_time:
        wh = "When"
    elif clause.pobj_place:
        wh = "Where"
    else:
        return None
    aux = _do_support(clause.verb_tag)
    if aux is None:
        return None
    words = [wh, aux, clause.subject, clause.verb_lemma]
    slots = [clause.subject, clause.verb_lemma, clause.prep, clause.pobj]
    if clause.obj:
        words.append(clause.obj)
        slots.append(clause.obj)
    return _pair(clause.sentence, " ".join(words) + "?", clause.sentence, slots)


def _statement(clause: Clause) -> Pair | None:
    if not clause.noun:
        return None
    return _pair(
        clause.sentence,
        f"Say a sentence about {clause.noun}.",
        clause.sentence,
        [clause.noun],
    )


def _parts(clause: Clause) -> Pair | None:
    if not clause.noun or not clause.verb or not clause.prep:
        return None
    assistant = (
        f"The noun is {clause.noun}. "
        f"The verb is {clause.verb}. "
        f"The preposition is {clause.prep}."
    )
    return _pair(
        clause.sentence,
        "Name the noun, the verb, and the preposition.",
        assistant,
        [clause.noun, clause.verb, clause.prep],
    )


def render_clause(clause: Clause) -> list[Pair]:
    pairs = []
    for build in (_subject_question, _object_question, _prep_question, _statement, _parts):
        pair = build(clause)
        if pair is not None:
            pairs.append(pair)
    return pairs


def dedupe(pairs: list[Pair]) -> list[Pair]:
    seen: set[str] = set()
    kept: list[Pair] = []
    for pair in pairs:
        key = " ".join(pair.user.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(pair)
    return kept


def render(clauses: list[Clause]) -> list[Pair]:
    pairs: list[Pair] = []
    for clause in clauses:
        pairs.extend(render_clause(clause))
    return dedupe(pairs)
