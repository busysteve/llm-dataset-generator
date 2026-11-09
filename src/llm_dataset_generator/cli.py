"""Command line for llm-dataset-generator."""

import argparse
import sys
from pathlib import Path

from llm_dataset_generator.extract import extract
from llm_dataset_generator.render import render
from llm_dataset_generator.write import write_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-dataset-generator")
    parser.add_argument("--input", required=True, help="UTF-8 prose document")
    parser.add_argument("--out", required=True, help="empty directory to create")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser


def _fail(message: str, code: int = 2) -> int:
    print(message, file=sys.stderr)
    return code


def _read_input(path: Path) -> str | int:
    if not path.is_file():
        return _fail(str(path))
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _fail("encoding")
    if not text.strip():
        return _fail("empty")
    return text


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_chars < 1:
        return _fail("max-chars")
    if args.max_pairs < 0:
        return _fail("max-pairs")
    loaded = _read_input(Path(args.input))
    if isinstance(loaded, int):
        return loaded
    try:
        import spacy
        nlp = spacy.load(args.model)
    except OSError:
        return _fail("model")
    pairs = render(extract(nlp, loaded, args.max_chars))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        return _fail("no pairs", 3)
    try:
        list_path = write_dataset(pairs, Path(args.out))
    except FileExistsError:
        return _fail("out")
    print(f"{len(pairs)} {list_path}")
    return 0


def main() -> None:
    raise SystemExit(run())
