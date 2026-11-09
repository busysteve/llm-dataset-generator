# llm-dataset-generator

Read a UTF-8 prose document and write `User:` / `Assistant:` exchanges for [busysteve/llm](https://github.com/busysteve/llm).

The tool parses each sentence with spaCy. It keeps the noun (subject or first noun), the finite verb, and up to three prepositional phrases. Those slots become questions and statements. The assistant answer for a question is the source sentence, so the reply stays the words that were in the document.

This repository does not train a model and does not run a conversation. `train_books` trains on the files. `chat` is the conversation program.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

`en_core_web_sm` is downloaded separately. It supplies the tagger, parser, lemmatizer, and named-entity labels used to choose Who, When, and Where.

## Generate

```bash
llm-dataset-generator --input examples/harbor.txt --out out
```

`out/exchanges/0001.txt` is one exchange:

```
User: Who rings the bell at dawn?
Assistant: The keeper rings the bell at dawn.
```

`out/exchanges.txt` lists `exchanges/0001.txt` and the rest. Paths are relative to the list file.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | UTF-8 document |
| `--out` | required | directory to create; must be missing or empty |
| `--model` | `en_core_web_sm` | spaCy model name |
| `--max-chars` | 400 | skip a sentence longer than this |
| `--max-pairs` | 0 | 0 writes every pair; a positive number stops after that many files |

Exit 0 prints the file count and the list path. Exit 2 is a bad input, a missing model, or an output directory that already has files. Exit 3 means the document produced no pairs, and nothing is written.

A sentence that contains `User:` or `Assistant:` is skipped. Questions whose slot text is not in the sentence are dropped. The same user line is kept once, from the first sentence that produced it.

## What becomes a question

For a finite root verb the tool takes:

- the subject noun chunk
- the object noun chunk, when there is one
- each preposition and its object, up to three

Who is used when the subject is a proper noun, a person name, or a common role noun such as keeper, cook, or sailor. Other nouns use What. When and Where are used only for time and place objects. Other prepositions appear in the subject question and in one grammar line:

```
User: Name the noun, the verb, and the preposition.
Assistant: The noun is keeper. The verb is rings. The preposition is at.
```

That grammar line is written once. Later sentences still add their own who, what, when, where, and "say a sentence" lines.

## Train

From a build of `llm`, point `train_books` at the list. One file is one exchange, and eos appended after the file ends the assistant turn.

```bash
./build/train_books --files /path/to/out/exchanges.txt --out chat-run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 5000
```

Use `--preset llama-tiny` when the checkpoint should be the block `gguf llama` exports. A short document teaches the turn shape. A usable chat model needs many sentences and a full training run.

```bash
./build/chat --model chat-run
```

Type only the question at `you>`. The program adds `User:` and `Assistant:`.

## Tests

```bash
pytest
```

`tests/test_render.py` and `tests/test_write.py` do not need the spaCy model. `tests/test_extract.py` runs when `en_core_web_sm` is installed.

## License

[CC0 1.0 Universal](LICENSE).
