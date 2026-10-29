# Cataloguing profiles

A profile decides **how one family of documents is described**: the persona and
domain rules given to the model, the metadata schema it must fill, the document
type vocabulary, how much of the document it gets to see, and which reading
strategy to use for long texts.

This matters because retrieval here is not similarity — it is an agent reading
descriptions and choosing. A description written by a generic cataloguer ("a
document about an agreement between two companies") is not enough to pick the
right contract out of four hundred. One written by a legal cataloguer ("supply
agreement between ACME and Contoso, effective 2024-01-01, auto-renewing, EUR
250,000 annual cap, penalty clause 12.3") is.

```bash
doccat profiles list              # what exists, how it routes
doccat profiles show legal        # the exact prompt it produces
doccat profiles init mydomain     # scaffold your own
doccat profiles validate --profiles-dir ./profiles
```

On a repository you do not know yet, `doccat optimize` drafts the profile for
you by reading a sample of it — see [OPTIMIZE.md](OPTIMIZE.md).

## The built-in profiles

| Profile | Priority | For | Adds |
|---|---|---|---|
| `generic` | 0 | anything (the fallback) | the core schema |
| `long_document` | 10 | files over ~2 MB | map-reduce over the whole text |
| `intranet` | 30 | internal policies, forms, circulars, HR | `applies_to`, `responsible_unit`, `forms_and_systems`, `deadlines`, `supersedes` |
| `technical` | 35 | manuals, runbooks, procedures, API docs | `systems`, `procedures`, `prerequisites`, `audience`, `error_codes` |
| `academic` | 35 | papers, theses, preprints | `authors`, `affiliations`, `venue`, `doi`, `research_question`, `methods`, `findings`, `limitations` |
| `minutes` | 35 | meeting minutes, agendas, correspondence | `meeting_date`, `attendees`, `decisions`, `action_items`, `open_issues`, `follow_up_date` |
| `legal` | 40 | contracts, tenders, regulations | `parties`, `effective_date`, `expiry_date`, `governing_law`, `monetary_values`, `obligations`, `key_clauses`, `risk_flags`, **`relations`** (graph-enabled) |
| `financial` | 40 | invoices, budgets, statements | `issuer`, `recipient`, `document_number`, `issue_date`, `due_date`, `currency`, `total_amount`, `line_items`, `fiscal_period` |
| `parliamentary` | 45 | plenary and committee proceedings | `chamber`, `session_date`, `session_number`, `agenda_items`, `speakers`, `parties`, `motions_and_votes`, `decisions`, **`relations`** (graph-enabled, dedicated pass) |

Every profile inherits the core schema (`title`, `summary`, `description`,
`doc_type`, `language`, `keywords`, `topics`, `entities`,
`questions_answered`), so one catalog can mix document families and the MCP
server's search works the same across all of them.

`legal` and `parliamentary` also extract **relations** between entities —
who supplies/guarantees/indemnifies whom, who proposed/voted on what — for
building a corpus graph (`doccat graph build`) queryable through the MCP
server's `graph_*` tools. See [GRAPH.md](GRAPH.md).

## Choosing a profile

Three ways, in order of precedence:

1. **Forced** — `--profile legal` (or `DOCCAT_PROFILE=legal`) applies one
   profile to every document. Documents already catalogued under a *different*
   profile are re-catalogued, so switching profiles does not need `--force`.
2. **Routed** — each profile's `match` block claims documents. The
   highest-priority profile whose rules hold wins.
3. **Fallback** — `generic`.

`--dry-run` shows the routing for a whole corpus without calling the LLM:

```bash
doccat index --root /data/documents --dry-run
```

```
Dry run — 8451 document(s) would be catalogued:
    generic          4102
    legal            1980
    technical        1233
    financial         874
    long_document     262
```

Rules that depend on document *content* (`content_regex`) cannot be evaluated in
a dry run, since nothing has been extracted yet; the command says so when any
loaded profile uses one.

## Writing a profile

A profile is a YAML file. Put it in a directory and pass `--profiles-dir DIR`
(repeatable; `DOCCAT_PROFILES_DIR` takes a `:`-separated list), or name a single
file with `--profile-file FILE`. A user profile that reuses a built-in name
replaces it entirely.

```yaml
name: clinical_trial              # required; also the --profile value
description: Clinical trial protocols and reports.
extends: academic                 # default: generic
priority: 70                      # higher wins when several profiles match

match:                            # omit to make the profile --profile-only
  any_of:
    - path_globs: ["**/trials/**"]
    - filename_regex: "(NCT[0-9]{8}|protocol)"

excerpt:
  strategy: head_tail
  head_chars: 10000
  tail_chars: 3000

llm:
  max_tokens: 6000

prompt:
  role: >-
    You are cataloguing clinical trial documentation for a systematic review team.
  guidance: >-
    Never infer an endpoint, a sample size or a phase that is not explicitly stated.
  doc_types: [protocol, statistical_analysis_plan, clinical_study_report, other]
  fields:
    trial_id: registry identifier (NCT..., EudraCT...), or empty
    phase: "trial phase: I, II, III, IV, or empty"
    primary_endpoints:
      type: list
      description: primary endpoints exactly as defined in the protocol
```

### Top-level keys

| Key | Meaning |
|---|---|
| `name` | Profile name; defaults to the file stem |
| `description` | One line, shown by `doccat profiles list` |
| `extends` | Parent profile to inherit from (default `generic`) |
| `priority` | Routing precedence; higher wins |
| `match` | Routing rules (see below) |
| `excerpt` | How much of the document the model sees |
| `llm` | Per-profile model/sampling overrides |
| `prompt` | Persona, rules, vocabulary and schema |
| `remove_fields` | List of inherited field names to drop |
| `graph` | Turns on relation extraction; entity/relation vocabulary (see below) |

Unknown keys are a hard error, at every level — a typo in a profile is a bug you
want at load time, not a silently missing field across ten thousand documents.

### Inheritance

`extends` deep-merges the child onto the parent: nested mappings merge, scalars
and lists replace. So a child that states only `prompt.guidance` keeps the whole
parent schema and changes one paragraph, while a child that states
`prompt.doc_types` replaces the vocabulary outright.

Two keys are never inherited: `match` and `remove_fields`. Routing is a decision
of the profile that states it — otherwise a child of `legal` would silently
claim every contract in the corpus.

### `match` — routing rules

Every rule stated at one level must hold (AND). `any_of` holds a list of nested
rule sets of which at least one must hold (OR) — that is how a profile says "the
folder *or* the file name tells me this is mine".

| Rule | Matches against |
|---|---|
| `path_globs` | the catalog path (`folder/file.pdf`, or `dump.json#key`), also tried with a leading `/` and against the bare file name |
| `filename_regex` | the file name, case-insensitive |
| `content_regex` | the first 4,000 extracted characters, case-insensitive |
| `filetypes` | `pdf`, `docx`, `json_record`, ... |
| `source_kinds` | `files` or `json` |
| `metadata_equals` | a JSON record's field, compared case-insensitively |
| `metadata_regex` | a JSON record's field, by regex |
| `min_bytes` / `max_bytes` | file size |

```yaml
# A folder OR a file name is enough, but it must be a PDF.
match:
  filetypes: [pdf]
  any_of:
    - path_globs: ["**/contracts/**"]
    - filename_regex: "(contrato|agreement)"
```

```yaml
# JSON dumps: route on a record's own metadata.
match:
  metadata_equals: { orga: Senado }
```

```yaml
# Last resort: look inside the document.
match:
  content_regex: "CONFIDENTIAL — ATTORNEY WORK PRODUCT"
```

### `excerpt` — what the model sees

| Strategy | Behaviour | Good for |
|---|---|---|
| `head_tail` (default) | first `head_chars` + last `tail_chars` | most documents: title page, abstract, conclusions |
| `head` | first `head_chars` only | documents that front-load everything |
| `full` | whole text up to `max_chars` | short documents (minutes, letters, emails) |
| `sampled` | head + `samples` evenly spaced windows + tail, capped at `max_chars` | long uniform texts (transcripts, registers) where the middle matters |
| `map_reduce` | chunk the text, take notes on each chunk, then catalogue from the notes | very long documents whose summary must cover the whole span |

Keys: `head_chars`, `tail_chars`, `max_chars`, `samples`, and for map-reduce
`chunk_chars`, `chunk_overlap`, `max_chunks`.

`map_reduce` costs `chunks + 1` LLM calls per document instead of one, so it is
worth reserving for documents that need it — which is exactly what the
`long_document` profile's `min_bytes: 2000000` rule does. Short texts routed to
a map-reduce profile fall back to a single call automatically.

### `llm` — per-profile overrides

```yaml
llm:
  model: a-bigger-model-for-this-domain   # overrides --model for this profile
  temperature: 0.0
  max_tokens: 8000
  think: true                              # keep reasoning mode on
  ingestion: upload                        # parse | upload | auto; overrides --mode
```

`ingestion: upload` sends the raw file to a multimodal model instead of
extracting text locally — the way to catalogue scanned PDFs without a separate
OCR step.

### `prompt` — persona, rules and schema

| Key | Meaning |
|---|---|
| `role` | Who the model is cataloguing as |
| `instructions` | Ground rules, inherited from `generic` unless overridden |
| `guidance` | Domain-specific rules, appended after the schema |
| `doc_types` | Controlled vocabulary; `{doc_types}` expands to the list |
| `fields` | The metadata schema (see below) |
| `output_language` | Force free-text fields into one language regardless of the document's |
| `map_instructions` | Prompt for the map phase of `map_reduce` |
| `reduce_instructions` | Prompt for the reduce phase |

`fields` maps a field name to either a description string, or a mapping:

```yaml
fields:
  venue: journal, conference or repository, with year, or empty   # string
  authors:
    type: list
    description: authors in the order printed
  entities:
    type: object
    fields:
      people: { type: list, description: named persons }
```

Types are `string` (default), `list`, `object`, `integer`, `number`, `boolean`.
They are used twice: to render the JSON skeleton shown to the model, and to
coerce what comes back — a model that returns `"a, b, c"` for a list field gets
it split into `["a", "b", "c"]` rather than producing a broken record.

A `list` field can hold structured elements instead of scalars with
`items: object` — a list of relations, each with its own `subject`/`relation`/
`object` is exactly this shape, and is how `graph:` (below) defines its
`relations` field automatically:

```yaml
fields:
  relations:
    type: list
    items: object
    fields:                      # the schema of EACH element, not of the list itself
      subject: the first entity
      relation: short label for the relationship
      object: the second entity
```

Core fields the profile declares but the model left empty are recorded in
`incomplete_fields` on the record, so a systematically weak field is visible:

```bash
doccat stats --catalog catalog.jsonl | grep incomplete
```

### `graph` — relations between entities

```yaml
graph:
  entity_types: [organization, person, role]   # vocabulary hint, not an enforced enum
  relation_types: [supplies_to, reports_to, party_to]
  dedicated_pass: false      # true = a second, focused LLM call just for relations
  min_confidence: 0.0        # drop relations the model itself flagged as weak
```

Declaring `graph:` (even empty) turns relation extraction on and gives the
profile a `relations` field for free, the same way every profile already gets
`entities` for free from `generic`. Absent `graph:`, nothing changes. Unlike
`match:`, `graph:` **inherits** — a profile extending an already graph-enabled
one keeps that config unless it states its own, since there's no routing
conflict to avoid here. Full reference, including what gets built from these
relations and how an agent queries it: [GRAPH.md](GRAPH.md).

## How profiles reach the agent

Profile fields are first-class catalog fields. The MCP server:

* includes them in the keyword search blob — a new profile's fields become
  searchable with no server change;
* reports them per profile in `catalog_overview` under `profile_fields`, so an
  agent discovers at runtime that contracts carry `parties` and transcripts
  carry `speakers`;
* exposes `search_by_field`, which queries one of them directly;
* accepts a `profile` filter on `search_documents` and `list_documents`.

```
catalog_overview()
  → profile_fields: {"legal": ["expiry_date", "parties", ...], ...}
search_by_field(field="parties", value="ACME")
  → the three contracts ACME is a party to
read_document(doc_id=...)
  → the clauses themselves
```

## Re-cataloguing after a profile change

A record stores the profile it was built with. Naming a profile explicitly
re-catalogues anything indexed under a different one:

```bash
doccat index --root /data/documents --profile legal --include "contracts/*"
```

Changing a profile's *content* (its prompt or schema) does not by itself trigger
re-indexing — the change key is the document's content hash. Re-run the affected
subset with `--force`:

```bash
doccat index --root /data/documents --include "contracts/*" --force
```
