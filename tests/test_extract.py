from pathlib import Path

import pytest

from llm_dataset_generator.cli import run
from llm_dataset_generator.extract import collapse, ensure_period, extract

spacy = pytest.importorskip("spacy")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm is not installed", allow_module_level=True)


def test_collapse_and_period():
    assert collapse("The   keeper\nrings") == "The keeper rings"
    assert ensure_period("The keeper rings the bell") == "The keeper rings the bell."
    assert ensure_period("Does he?") == "Does he?"


def test_keeper_clause_slots():
    clauses = extract(nlp, "The keeper rings the bell at dawn.", 400)
    assert clauses
    clause = clauses[0]
    assert clause.subject == "the keeper"
    assert clause.subject_proper is True
    assert clause.verb == "rings"
    assert clause.verb_lemma == "ring"
    assert clause.obj == "the bell"
    assert clause.prep == "at"
    assert clause.pobj == "dawn"
    assert clause.pobj_time is True
    assert clause.noun == "keeper"


def test_long_sentence_is_skipped():
    text = "The keeper rings the bell at dawn."
    assert extract(nlp, text, 10) == []


def test_cli_writes_the_worked_example(tmp_path: Path, capsys):
    source = tmp_path / "harbor.txt"
    source.write_text("The keeper rings the bell at dawn.\n", encoding="utf-8")
    out = tmp_path / "out"
    code = run(["--input", str(source), "--out", str(out)])
    assert code == 0
    listed = (out / "exchanges.txt").read_text(encoding="utf-8").splitlines()
    first = (out / listed[0]).read_text(encoding="utf-8")
    assert first == (
        "User: Who rings the bell at dawn?\n"
        "Assistant: The keeper rings the bell at dawn.\n"
    )
    assert "exchanges.txt" in capsys.readouterr().out


def test_cli_max_text_length_drops_the_rest(tmp_path: Path):
    first = "The keeper rings the bell at dawn.\n"
    source = tmp_path / "two.txt"
    source.write_text(first + "The cook sets the bread in the kitchen.\n", encoding="utf-8")
    out = tmp_path / "out"
    code = run([
        "--input", str(source),
        "--out", str(out),
        "--max-text-length", str(len(first)),
    ])
    assert code == 0
    body = "".join(path.read_text(encoding="utf-8") for path in (out / "exchanges").iterdir())
    assert "keeper" in body
    assert "cook" not in body


def test_cli_no_pairs(tmp_path: Path, capsys):
    source = tmp_path / "fragment.txt"
    source.write_text("...\n", encoding="utf-8")
    code = run(["--input", str(source), "--out", str(tmp_path / "out")])
    assert code == 3
    assert capsys.readouterr().err.strip() == "no pairs"
    assert not (tmp_path / "out").exists()
