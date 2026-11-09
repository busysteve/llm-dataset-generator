from pathlib import Path

from llm_dataset_generator.cli import run


def test_missing_input_prints_the_path(capsys):
    code = run(["--input", "no-such-document.txt", "--out", "out"])
    assert code == 2
    assert capsys.readouterr().err.strip() == "no-such-document.txt"


def test_empty_input(tmp_path: Path, capsys):
    source = tmp_path / "empty.txt"
    source.write_text(" \n\t", encoding="utf-8")
    code = run(["--input", str(source), "--out", str(tmp_path / "out")])
    assert code == 2
    assert capsys.readouterr().err.strip() == "empty"


def test_bad_encoding(tmp_path: Path, capsys):
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff\xfe not utf-8")
    code = run(["--input", str(source), "--out", str(tmp_path / "out")])
    assert code == 2
    assert capsys.readouterr().err.strip() == "encoding"


def test_rejects_nonpositive_max_chars(tmp_path: Path, capsys):
    source = tmp_path / "doc.txt"
    source.write_text("The keeper rings the bell.\n", encoding="utf-8")
    code = run(["--input", str(source), "--out", str(tmp_path / "out"), "--max-chars", "0"])
    assert code == 2
    assert capsys.readouterr().err.strip() == "max-chars"
