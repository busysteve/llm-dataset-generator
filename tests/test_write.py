from pathlib import Path

import pytest

from llm_dataset_generator.render import Pair
from llm_dataset_generator.write import exchange_width, write_dataset


def test_width_stays_four_until_10000():
    assert exchange_width(1) == 4
    assert exchange_width(9999) == 4
    assert exchange_width(10000) == 5


def test_writes_exact_exchange_bytes(tmp_path: Path):
    pairs = [
        Pair("Who rings the bell?", "The keeper rings the bell."),
        Pair("Say a sentence about keeper.", "The keeper rings the bell."),
    ]
    list_path = write_dataset(pairs, tmp_path / "out")
    assert list_path.read_bytes() == b"exchanges/0001.txt\nexchanges/0002.txt\n"
    first = (tmp_path / "out" / "exchanges" / "0001.txt").read_bytes()
    assert first == b"User: Who rings the bell?\nAssistant: The keeper rings the bell.\n"


def test_refuses_nonempty_output(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_dataset([Pair("Who rings?", "The keeper rings.")], out)
