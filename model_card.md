# Model Card — Research for Humans

**Venue:** International Society for Learning Sciences (ISLS) 2026 Conference, UC Irvine  
**Project:** Research for Humans  
**Team:** Emily Reardon, Lucius Von Joo, OreOluwa Badaki, Chris Moffett  
Digital Futures Institute, Teachers College, Columbia University  
Contact: er3343@tc.columbia.edu

---

## What this is

Research for Humans is an interactive multimodal installation using social practice art to address a global epistemic crisis. Trust in reality wavers as traditional institutions lose their footing, while technology accelerates the need for reinvention. What is truth, how do we know it, and how might we design ways forward?

Inspired by centuries of automata, the work reflects our enduring fascination with what it means to be alive and the uneasy balance between human vitality and machine logic. Visitors approach a roving cart overseen by an animatronic peddler — part mystic, part minstrel — and converse by talking or typing. Responses drawn from published research are printed in real-time on cards: one to keep, one added to a growing repository.

Social practice art transforms research into communal inquiry. Here, academic knowledge becomes something felt, embodied, and carried home — a provocation toward braver thinking and social action.

---

## The model

**Language model:** OLMo 3 7B  
**Developer:** Allen Institute for AI (AI2) — nonprofit, fully open-weight  
**Running via:** Ollama, locally on a MacBook Pro. Nothing leaves the device.  
**Embeddings:** nomic-embed-text (open-source, local)  
**Retrieval:** ChromaDB, local vector store  
**OLMo 3 training data:** Dolma 3 — a curated open dataset of web text, academic papers, code, and books. Details at allenai.org/olmo3.

We explicitly chose not to use: OpenAI, Anthropic, Google, Meta (Llama), Microsoft (Phi), or Alibaba (Qwen). Those are products built for commercial return. OLMo is built for research and public benefit.

---

## Intended use

**Intended context:** Public-facing art installation at an academic conference. Visitors ask questions; the system responds from the ISLS corpus and prints a card.  
**Intended users:** Conference attendees engaging with questions about learning, knowledge, and education.  
**Out-of-scope uses:** Clinical, legal, or diagnostic advice; surveillance; any context where a fabricated citation or hallucinated fact could cause harm. Not designed for autonomous decision-making or deployment outside a facilitated setting.

---

## The corpus

**Source:** repository.isls.org — open-access ISLS conference proceedings  
**Size:** 7,463 papers  
**Coverage:** CSCL, ICLS, and related ISLS venues  
**Ingestion:** Full PDF text extracted and indexed — not just abstracts. Each paper is chunked and embedded so retrieval can surface specific findings, methods, and cited works from inside the paper, not only its summary.  
**Method:** Retrieval-augmented generation (RAG). The model does not generate from memory — it retrieves relevant passages from the corpus and responds from that evidence.

---

## Response format

Responses are written in an oracle card mode. The card surfaces a tension from inside the corpus — a specific cross-paper relational friction grounded in two attributable poles, something not legible from within any single paper. It does not summarize. It does not report. It opens a problem.

Each card has three parts: an opening claim or aporia drawn from the corpus; a body that uses specific evidence from retrieved papers, framed speculatively; and a closing question that opens onto a conversation the reader generates. The card is self-contained — legible without knowledge of the question that prompted it.

Up to three source papers are cited per card, with QR codes linking to each.

---

## Factors and performance

Response quality varies by:
- **Domain proximity:** Questions closely related to learning science yield better-grounded responses than peripheral topics.
- **Language:** Nearly all corpus papers are in English; non-English questions may retrieve less relevant results.
- **Question type:** Abstract or values-based questions perform differently than specific methodological ones. Tensions are organized by four relation types: opposition, double-edge, incompatible operationalization, and convergence.

No quantitative evaluation has been conducted. This is an art installation, not a production AI system. Responses should be read as provocations, not findings.

---

## Ethical considerations

**Corpus consent:** Individual authors did not consent to inclusion. The corpus is open-access under ISLS terms, but individual permissioning was not sought. Removal requests: er3343@tc.columbia.edu.

**Known biases:**
- Anglophone bias — most papers are in English
- Conference-going, university-affiliated scholars are overrepresented
- Open-access only — paywalled work is absent
- The ISLS community reflects its own demographic and disciplinary skews

**Hallucination risk:** RAG substantially reduces but does not eliminate fabricated content. The system will not generate a response if the corpus returns no relevant results. Citations are drawn from retrieved metadata, not generated by the model.

**Data privacy:** No visitor input is stored or transmitted. The model runs entirely locally; nothing leaves the device.

**Power and representation:** The corpus reflects who publishes in Western academic conferences — who gets to travel, who gets institutional support, whose language dominates. Naming this is part of the work, not a footnote.

---

## Design decisions and values

**Radical attribution.** Every response names the author(s) of the work it draws from, with full first names. Researchers are people, not data points.

**Evidence-only.** If the corpus contains nothing genuinely relevant to a question, the system says so. It does not generate without grounding.

**Situated knowing.** The corpus has known biases. We name them. We do not claim completeness or neutrality.

**Care over efficiency.** Responses are designed for conversation, not resolution. The card is a prompt, not an answer. Visitors wear their cards. The installation is a social surface.

**Points away from technology toward humans.** The oracle card format is designed to open problems that the people in the room are better positioned to resolve than any model.

**Seventh-generation thinking.** We built this for a specific community, at a specific moment, with specific values. Those choices are documented here so others can see them, critique them, and make different ones.

---

## What the system will not do

- Generate a response without corpus evidence
- Display errors to visitors (errors log server-side only)
- Fabricate citations
- Use commercial AI infrastructure

---

## Transparency

This model card travels with the installation. The stack is fully open-source and reproducible. We are happy to share code, architecture, and process with anyone who wants to build something like this for their own community.

What most people know as AI is built via extraction by corporations, optimized for commercial return. That shapes what it knows, whose knowledge it centers, and what futures it makes imaginable. Different builders make different things; researchers, educators, artists, community builders: be brave.
