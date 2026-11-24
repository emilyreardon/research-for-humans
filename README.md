# Research for Humans

An automaton you can ask a question. It searches all 7,463 open-access papers from the International Society of the Learning Sciences, writes a short reply grounded in what it finds, and prints it twice as label stickers: one to wear, one added to the installation. Shown at the ISLS 2026 Gallery, UC Irvine.

Project page and AI model card: https://emilyreardon.org/research-for-humans.html

## What's here

| | |
|---|---|
| `app.py` | The automaton's server: retrieval, reply, printing, kiosk screen |
| `precompute_abstracts.py`, `precompute_enriched.py` | Prepare paper summaries and references ahead of time |
| `tools/step3_extract_tensions.py` | Finds the tensions between findings that replies are built around (`data/tensions.json`) |
| `Research for Humans … .txt` | Voice, format and search settings the app reads at start-up |
| `prompts/` | Prompt templates |
| `card-validator-rules.json` | Checks each reply before it prints |
| `circuitpython/` | Code for the automaton's lights |
| `launch_kiosk.sh` | Starts the server and opens the kiosk screen |

The corpus is not included, and neither is the script that collects it: the papers are open access, but this project won't make bulk downloading them for AI training easier. Visitor questions are never stored in this repository.

## How it runs

Entirely on one laptop, offline: [Ollama](https://ollama.com) with OLMo 3 from the nonprofit Allen Institute for AI, `nomic-embed-text` embeddings, and a local ChromaDB index.

## Credits

With Lucius Von Joo, Chris Moffett and OreOluwa Badaki, Digital Futures Institute, Teachers College, Columbia University. Emily Reardon: concept, corpus, Ollama backend, voice. Code written with help from Claude (Anthropic).

Code: MIT License (see `LICENSE`). The papers belong to their authors and are open access via ISLS.
