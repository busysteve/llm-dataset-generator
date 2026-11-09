"""Write one exchange per file plus the path list train_books reads."""

from pathlib import Path

from llm_dataset_generator.render import Pair


def exchange_width(count: int) -> int:
    return max(4, len(str(count)))


def write_dataset(pairs: list[Pair], out: Path) -> Path:
    if not pairs:
        raise ValueError("no pairs")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("out")
    out.mkdir(parents=True, exist_ok=True)
    width = exchange_width(len(pairs))
    folder = out / "exchanges"
    folder.mkdir()
    lines: list[str] = []
    for index, pair in enumerate(pairs, start=1):
        name = f"{index:0{width}d}.txt"
        body = f"User: {pair.user}\nAssistant: {pair.assistant}\n"
        (folder / name).write_bytes(body.encode("utf-8"))
        lines.append(f"exchanges/{name}\n")
    list_path = out / "exchanges.txt"
    list_path.write_bytes("".join(lines).encode("utf-8"))
    return list_path
