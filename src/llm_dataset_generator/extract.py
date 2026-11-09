"""Read a spaCy document into clauses of noun, verb, and preposition slots."""

import spacy
from spacy.tokens import Doc, Span, Token

from llm_dataset_generator.render import Clause

FINITE = {"VB", "VBP", "VBZ", "VBD"}
TIME_LEMMAS = frozenset({
    "dawn", "dusk", "morning", "evening", "night", "noon", "midnight",
    "hour", "day", "today", "tomorrow", "yesterday", "time", "tide",
    "week", "month", "year",
})
PLACE_LEMMAS = frozenset({
    "harbor", "harbour", "house", "home", "room", "road", "street",
    "town", "city", "sea", "shore", "hill", "garden", "kitchen",
})
# en_core_web_sm often misses role nouns such as "keeper". Who-questions
# need those heads marked as people or the worked example becomes What.
PERSON_ROLES = frozenset({
    "keeper", "cook", "captain", "sailor", "man", "woman", "boy", "girl",
    "child", "person", "people", "friend", "neighbor", "neighbour",
    "teacher", "doctor", "king", "queen", "pilot", "driver", "farmer",
    "baker", "guard", "nurse", "writer", "poet", "parent", "father",
    "mother", "brother", "sister", "son", "daughter",
})
_DETS = {"the", "a", "an"}


def collapse(text: str) -> str:
    return " ".join(text.split())


def ensure_period(text: str) -> str:
    if text and text[-1] not in ".!?":
        return text + "."
    return text


def _span_for(token: Token) -> Span | Token:
    for chunk in token.sent.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk
    return token


def _tokens_of(span: Span | Token) -> list[Token]:
    if isinstance(span, Token):
        return [span]
    return list(span)


def _chunk_text(token: Token) -> str:
    return collapse(_span_for(token).text)


def _embed_subject(text: str) -> str:
    """Lowercase a sentence-initial determiner so it can sit inside a question."""
    parts = text.split()
    if parts and parts[0] in {"The", "A", "An"}:
        parts[0] = parts[0].lower()
    return " ".join(parts)


def _strip_det(text: str) -> str:
    parts = text.split()
    if parts and parts[0].casefold() in _DETS:
        parts = parts[1:]
    return " ".join(parts)


def _lemma(token: Token) -> str:
    return token.lemma_.lower()


def _is_time(span: Span | Token) -> bool:
    tokens = _tokens_of(span)
    if any(token.ent_type_ in {"DATE", "TIME"} for token in tokens):
        return True
    return any(_lemma(token) in TIME_LEMMAS for token in tokens)


def _is_place(span: Span | Token) -> bool:
    tokens = _tokens_of(span)
    if any(token.ent_type_ in {"GPE", "LOC", "FAC"} for token in tokens):
        return True
    return any(_lemma(token) in PLACE_LEMMAS for token in tokens)


def _classify(token: Token) -> tuple[bool, bool]:
    span = _span_for(token)
    if _is_time(span):
        return True, False
    return False, _is_place(span)


def _is_proper(token: Token) -> bool:
    span = _span_for(token)
    tokens = _tokens_of(span)
    root = span.root if isinstance(span, Span) else token
    if root.pos_ == "PROPN" or any(item.ent_type_ == "PERSON" for item in tokens):
        return True
    return _lemma(root) in PERSON_ROLES


def _first(token: Token, labels: set[str]) -> Token | None:
    found = [
        child for child in token.children
        if child.dep_ in labels or child.dep_.split(":")[0] in labels
    ]
    if not found:
        return None
    return min(found, key=lambda child: child.i)


def _finite_verb(sent: Span) -> Token | None:
    roots = [token for token in sent if token.dep_ == "ROOT" and token.tag_ in FINITE]
    if not roots:
        return None
    return roots[0]


def _has_noun(sent: Span) -> bool:
    return any(token.pos_ in {"NOUN", "PROPN"} for token in sent)


def _noun_of(sent: Span, subject: Token | None) -> str:
    if subject is not None:
        return _strip_det(_chunk_text(subject))
    chunks = list(sent.noun_chunks)
    if chunks:
        return _strip_det(collapse(chunks[0].text))
    for token in sent:
        if token.pos_ in {"NOUN", "PROPN"}:
            return _strip_det(collapse(token.text))
    return ""


def _prep_phrases(verb: Token) -> list[tuple[str | None, str | None, bool, bool]]:
    phrases: list[tuple[str | None, str | None, bool, bool]] = []
    children = sorted(verb.children, key=lambda child: child.i)
    for child in children:
        label = child.dep_.split(":")[0]
        prep = ""
        pobj: Token | None = None
        if label == "prep":
            pobj = _first(child, {"pobj", "obj"})
            prep = collapse(child.text)
        elif label == "obl":
            case = _first(child, {"case"})
            if case is not None:
                prep = collapse(case.text)
                pobj = child
        if not prep or pobj is None:
            continue
        time, place = _classify(pobj)
        phrases.append((prep, _chunk_text(pobj), time, place))
        if len(phrases) == 3:
            break
    return phrases


def _clauses_for(sent: Span, sentence: str) -> list[Clause]:
    if not _has_noun(sent):
        return []
    verb = _finite_verb(sent)
    if verb is None:
        return []
    subject = _first(verb, {"nsubj", "nsubjpass"})
    obj = _first(verb, {"dobj", "obj", "attr"})
    noun = _noun_of(sent, subject)
    if not noun:
        return []
    subject_text = _embed_subject(_chunk_text(subject)) if subject is not None else None
    proper = _is_proper(subject) if subject is not None else False
    obj_text = _chunk_text(obj) if obj is not None else None
    phrases = _prep_phrases(verb)
    if not phrases:
        phrases = [(None, None, False, False)]
    clauses = []
    for prep, pobj, time, place in phrases:
        clauses.append(Clause(
            sentence=sentence,
            subject=subject_text,
            subject_proper=proper,
            verb=collapse(verb.text),
            verb_lemma=_lemma(verb),
            verb_tag=verb.tag_,
            obj=obj_text,
            prep=prep,
            pobj=pobj,
            pobj_time=time,
            pobj_place=place,
            noun=noun,
        ))
    return clauses


def extract(nlp: spacy.language.Language, text: str, max_chars: int) -> list[Clause]:
    # spaCy rejects len(text) > nlp.max_length. The default is 1_000_000.
    if len(text) > nlp.max_length:
        nlp.max_length = len(text)
    doc: Doc = nlp(text)
    clauses: list[Clause] = []
    for sent in doc.sents:
        raw = collapse(sent.text)
        if not raw or len(raw) > max_chars:
            continue
        if "User:" in raw or "Assistant:" in raw:
            continue
        clauses.extend(_clauses_for(sent, ensure_period(raw)))
    return clauses
