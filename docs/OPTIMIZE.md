# Designing a profile for a repository you don't know

Picking the right cataloguing profile by hand means knowing what is in the
repository — and the reason to catalogue a repository is usually that nobody
does. `doccat optimize` closes that loop: it reads a sample of the documents,
works out what the corpus is, and writes a profile tuned to it.

```bash
doccat optimize --root /data/documents -o profiles/mydomain.yaml
```

```
Phase 1/4 — sampling up to 12 document(s)
Phase 2/4 — analysing the sample (12 document(s), 68014 chars)
Phase 3/4 — drafting the profile
Drafted profile 'municipal_procurement': 5 extra field(s), strategy head_tail, extends legal
Phase 4/4 — refinement round 1/1: cataloguing 3 sample document(s)
Trial: {"documents_catalogued": 3, "always_empty": ["awarded_to", "bidders"], "no_variation": []}

Wrote profiles/municipal_procurement.yaml
  profile     municipal_procurement (extends legal)
  domain      public procurement
  families    tender specifications, award decisions, framework agreements
  strategy    head_tail  (The reference, budget and deadline sit in the header; the award sits at the end.)
  doc types   tender_specification, award_decision, framework_agreement, contract_modification, other
  new fields  tender_reference, contracting_authority, awarded_to, budget, submission_deadline
  inherited   parties, effective_date, expiry_date, governing_law, monetary_values (+3 more)  (from legal)
  sampled     12 document(s), 5 LLM call(s)
  refine      round 1 | always empty: awarded_to, bidders
              Dropped `bidders`: empty in every trial record. Kept the rest.

Review it, then use it:
  doccat profiles show municipal_procurement --profile-file profiles/municipal_procurement.yaml
  doccat index --root /data/documents --profiles-dir profiles --dry-run
```

The output is a YAML file you own and can edit — not a hidden prompt. Nothing is
applied automatically.

## The four phases

### 1. Sample

A stratified draw by (top-level folder, file type), round-robin across strata.
A corpus of 9,000 invoices and 40 contracts still puts a contract in front of
the agent — and the contracts are exactly the documents a generic profile
handles worst.

The draw is seeded, so re-running on an unchanged corpus reads the same
documents (`--seed` to change it). Documents with no extractable text are
skipped rather than wasting a slot on an empty sample.

### 2. Analyse

One LLM call over the whole sample. The prompt tells the model what the metadata
is *for* — that an agent will later choose documents from these records alone,
without a vector store — and asks it to decide what those records need to
contain. It returns the corpus summary, the document families present, a
persona, domain rules, a document-type vocabulary, the extra fields to extract,
which excerpt strategy suits these documents, and — when the sample actually
shows named parties relating to each other (who supplies whom, who reports to
whom) — an `entity_types`/`relation_types` vocabulary for a `graph:` block
(see [GRAPH.md](GRAPH.md)). The prompt explicitly tells it to leave both empty
when a graph wouldn't help: a self-contained manual or FAQ gains nothing from
one, and a forced vocabulary produces relations nobody asked for.

The instruction that does most of the work: a field is only worth proposing if
it would actually differ between documents in *this* corpus. A field that is
empty or identical everywhere costs tokens and tells a searching agent nothing.

### 3. Draft

The analysis is turned into a real profile YAML, and everything the model
proposed is sanitised rather than trusted:

| The model says | What happens |
|---|---|
| `profile_name: "My Weird Name!! "` | → `my_weird_name` |
| `extends: something_nonexistent` | → `generic` |
| `type: array` | → `list` (synonyms accepted); anything unrecognised → `string` |
| `strategy: telepathy` | → `head_tail`, with a warning |
| `head_chars: 9000000` | clamped to a sane range |
| `filename_regex: "(unclosed"` | dropped, with a warning |
| a field named `summary` | dropped — core fields are inherited, not redefined |
| a field with no name or no description | dropped |

The draft is then loaded through the normal profile machinery. **An invalid
draft is an error here**, not a file you discover is broken 10,000 documents
later.

The file it writes carries its own provenance in comments — the corpus summary,
the domain, the families found, why those fields, why that strategy — so a
reviewer months later can see what it was reasoning from.

### 4. Refine

The draft is used to catalogue a few real documents (`--trial-docs`, default 3).
The agent then sees its own output and revises the profile.

A mechanical check runs alongside, and is fed to the agent with the records:

- `always_empty` — fields no document filled in
- `often_empty` — filled for less than half
- `no_variation` — identical across every trial document
- `relations_without_matching_entities` — a `graph:`-enabled draft only: relations
  whose subject or object isn't among that same document's own extracted
  entities, a cheap sign the model is inventing connections rather than
  reporting ones actually in the text

Those are easier to spot by counting than by reading. In the run above, the
agent proposed `bidders`, saw it come back empty on every tender specification
(bids are not listed in the spec itself), and dropped it.

`--refine 0` skips the trial and stops at the draft — one LLM call in total.
`--refine 2` runs the loop twice. A refine round that fails, or that produces an
invalid profile, is logged and the previous good draft is kept.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--root DIR` | `paths.docs_root` | The repository to analyse |
| `--input FILE` | — | Analyse a JSON array dump instead (`--field`, `--limit`) |
| `--sample N` | `optimize.sample_size` (12) | Documents to read |
| `--refine N` | `optimize.refine_rounds` (1) | Critique-and-improve rounds |
| `--trial-docs N` | 3 | Documents catalogued for real per round |
| `--name NAME` | the agent's choice | Name for the profile |
| `--seed N` | 0 | Sampling seed |
| `--output`, `-o` | `<profiles-dir>/<name>.yaml` | Where to write it |
| `--print-only` | — | Print to stdout, write nothing |
| `--force-write` | — | Overwrite an existing file |
| `--include`, `--exclude` | — | Glob patterns limiting the sample |

Plus the usual `--base-url`, `--model`, `--api-key` and `--config`.

Cost is `1 + (trial_docs + 1) × refine_rounds` LLM calls — five by default.

## Using the result

```bash
doccat profiles show municipal_procurement --profile-file profiles/municipal_procurement.yaml
```

Read the prompt it produces. Then check the routing it proposed against your
actual corpus, without spending a token:

```bash
doccat index --root /data/documents --profiles-dir profiles --dry-run
```

Then a small real run before committing to the whole repository:

```bash
doccat index --root /data/documents --profiles-dir profiles --max-docs 50
```

```bash
doccat stats --catalog catalog.jsonl
```

`incomplete` in the stats, and `incomplete_fields` on individual records, show
fields the model kept leaving empty at full scale — the same signal the refine
phase uses, now measured over the real corpus. Edit the YAML and re-run the
affected subset with `--force`.

## When to use it, and when not to

**Use it** on an unfamiliar repository, on a domain none of the nine built-in
profiles fits, or as a starting point that is faster to edit than a blank file.

**Skip it** when a built-in profile already fits — `doccat profiles list`, then
`--dry-run` to see the routing. A hand-written profile from someone who knows
the domain beats a sampled one; `doccat profiles init` scaffolds an empty one.

**Run it more than once** on a heterogeneous repository. A corpus with distinct
families is better served by several narrow profiles than one broad one, since
routing picks between them per document:

```bash
doccat optimize --root /data/documents --include "legal/**" --name legal_house
```

```bash
doccat optimize --root /data/documents --include "hr/**" --name hr_house
```

## Always review it

The agent read a dozen documents out of thousands. It can miss a family absent
from the sample, propose a field that only exists in the documents it happened
to see, or write guidance that is subtly wrong for your domain. Treat the output
as a first draft from someone who skimmed the repository — useful, faster than
starting from nothing, and not a substitute for knowing your own documents.
