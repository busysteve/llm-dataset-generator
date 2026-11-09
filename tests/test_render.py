from llm_dataset_generator.render import Clause, render, render_clause

KEEPER = Clause(
    sentence="The keeper rings the bell at dawn.",
    subject="the keeper",
    subject_proper=True,
    verb="rings",
    verb_lemma="ring",
    verb_tag="VBZ",
    obj="the bell",
    prep="at",
    pobj="dawn",
    pobj_time=True,
    pobj_place=False,
    noun="keeper",
)


def test_keeper_templates():
    pairs = render_clause(KEEPER)
    assert [(pair.user, pair.assistant) for pair in pairs] == [
        (
            "Who rings the bell at dawn?",
            "The keeper rings the bell at dawn.",
        ),
        (
            "What does the keeper ring?",
            "The keeper rings the bell at dawn.",
        ),
        (
            "When does the keeper ring the bell?",
            "The keeper rings the bell at dawn.",
        ),
        (
            "Say a sentence about keeper.",
            "The keeper rings the bell at dawn.",
        ),
        (
            "Name the noun, the verb, and the preposition.",
            "The noun is keeper. The verb is rings. The preposition is at.",
        ),
    ]


def test_plural_go_uses_goes_and_omits_missing_object():
    clause = Clause(
        sentence="The boats go home.",
        subject="The boats",
        subject_proper=False,
        verb="go",
        verb_lemma="go",
        verb_tag="VBP",
        obj=None,
        prep=None,
        pobj=None,
        pobj_time=False,
        pobj_place=False,
        noun="boats",
    )
    users = [pair.user for pair in render_clause(clause)]
    assert users == [
        "What goes?",
        "Say a sentence about boats.",
    ]


def test_slot_missing_from_sentence_drops_that_question():
    clause = Clause(
        sentence="The keeper rings the bell at dawn.",
        subject="the keeper",
        subject_proper=True,
        verb="rings",
        verb_lemma="ring",
        verb_tag="VBZ",
        obj="the moon",
        prep="at",
        pobj="dawn",
        pobj_time=True,
        pobj_place=False,
        noun="keeper",
    )
    users = [pair.user for pair in render_clause(clause)]
    assert users == [
        "Say a sentence about keeper.",
        "Name the noun, the verb, and the preposition.",
    ]
    assert all("moon" not in user for user in users)


def test_where_and_deduped_parts_line():
    first = Clause(
        sentence="The cook works in the kitchen.",
        subject="the cook",
        subject_proper=True,
        verb="works",
        verb_lemma="work",
        verb_tag="VBZ",
        obj=None,
        prep="in",
        pobj="the kitchen",
        pobj_time=False,
        pobj_place=True,
        noun="cook",
    )
    second = Clause(
        sentence="The sailor waits in the harbor.",
        subject="the sailor",
        subject_proper=True,
        verb="waits",
        verb_lemma="wait",
        verb_tag="VBZ",
        obj=None,
        prep="in",
        pobj="the harbor",
        pobj_time=False,
        pobj_place=True,
        noun="sailor",
    )
    users = [pair.user for pair in render([first, second])]
    assert "Where does the cook work?" in users
    assert "Where does the sailor wait?" in users
    assert users.count("Name the noun, the verb, and the preposition.") == 1


def test_copula_skips_do_support():
    clause = Clause(
        sentence="The keeper is a sailor.",
        subject="the keeper",
        subject_proper=True,
        verb="is",
        verb_lemma="be",
        verb_tag="VBZ",
        obj="a sailor",
        prep=None,
        pobj=None,
        pobj_time=False,
        pobj_place=False,
        noun="keeper",
    )
    users = [pair.user for pair in render_clause(clause)]
    assert users == [
        "Who is a sailor?",
        "Say a sentence about keeper.",
    ]
