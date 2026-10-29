# Relation extraction and graph-based retrieval

Catalog records already carry `entities` (people, organizations, places,
dates) per document. A `graph:`-enabled profile also extracts `relations`
between those entities — who did what to whom, which document says so — and
`doccat graph build` consolidates every document's relations into one graph
across the whole corpus, queryable through five MCP tools.

```bash
doccat profiles show legal        # already graph-enabled — see its relations field
doccat index --root /data/documents --profile legal
doccat graph build --catalog catalog.jsonl --summarize
doccat serve --catalog catalog.jsonl    # graph_overview, graph_search, ... are live
```

## Where this sits relative to the rest of the system

Nothing about the core bet changes: retrieval is still an agent reading
descriptions and deciding, not a vector search. A graph here is not a
replacement for that — it's a second index an agent can consult when the
question is relational ("how are X and Y connected?", "who reports to whom?",
"what depends on what?") rather than topical ("which documents discuss X?").
`search_documents` / `search_by_field` stay the primary tools; the graph tools
are for the subset of questions where *structure between entities*, not
*content of one document*, is the answer. Every graph tool result carries the
`doc_id`s backing it, so the normal next step is still `read_document` — the
graph points at evidence, it never replaces reading it.

## 1. Extraction: what a profile produces per document

Relations are an open-vocabulary triple with a description, not an RDF-style
fixed predicate — a fixed predicate vocabulary fights LLM extraction (the
model invents labels you didn't define, or gets forced into one that half
fits), and it reintroduces exactly the cross-document-consistency problem
entity resolution (§3) already has to solve once:

```json
{"subject": "ACME Corp", "relation": "supplies_to", "object": "Contoso",
 "description": "ACME is the exclusive widget supplier to Contoso under the 2024 agreement",
 "strength": 0.9}
```

`relation` is a short, model-chosen label — useful for grouping and a quick
scan — but `description` is the fact an agent actually reads, the same way
`summary` is ground truth and `doc_type` is just a label.

### Turning it on for a profile

A `graph:` block, parallel to `match:`/`excerpt:`/`llm:`:

```yaml
graph:
  entity_types: [organization, person, role]   # vocabulary hint, not an enforced enum
  relation_types: [reports_to, supplies_to, party_to]
  dedicated_pass: false      # true = a second, focused LLM call just for relations
  min_confidence: 0.0        # drop relations the model itself flagged as weak
```

The profile gets a `relations` field for free the moment it declares `graph:`
— the same way every profile already gets `entities` for free from `generic`.
`entity_types`/`relation_types` fold into the prompt as vocabulary hints, never
an enforced enum (constraining the label is exactly the RDF brittleness this
format avoids). Absent `graph:`, nothing changes — a profile that doesn't
declare it extracts exactly what it did before.

Two built-ins ship with it already: `legal` (parties, who supplies/guarantees/
indemnifies whom) and `parliamentary` (who proposed/voted/spoke, with
`dedicated_pass: true` — see below). The other seven are untouched.

`graph:` **inherits** like `excerpt:`/`llm:` (unlike `match:`): a profile
extending an already graph-enabled one keeps that config unless it states its
own. `doccat optimize` proposes `entity_types`/`relation_types` too, when the
sample actually shows a network of named parties worth tracing — see
[OPTIMIZE.md](OPTIMIZE.md).

### Inline vs. dedicated pass

By default relations ride the document's one existing LLM call — the pipeline
makes exactly as many calls per document as it did before, because `relations`
is just a bigger JSON schema in the same completion. Set
`dedicated_pass: true` when a document family is dense enough that cramming
entities, relations and everything else into one completion starts to compete
for the model's attention (a 40-clause contract with a dozen cross-referenced
parties is the likely case): a second, focused call runs afterward, grounded
in the entities the first call already found, costing one extra call per
document for that profile only.

`min_confidence` drops relations below that `strength` after extraction — a
relation with no `strength` at all is never penalised, only ones the model
itself flagged as weak are dropped. Records a `relations_below_confidence`
count on the catalog record when anything was dropped.

### The schema engine underneath it

`relations` needs a **list of structured objects**
(subject/relation/object/description/strength), which required extending
`doccat/profiles.py`'s field schema beyond lists of scalars:

```yaml
fields:
  relations:
    type: list
    items: object            # list elements are structured, not scalar strings
    fields:                  # the same nested-fields mechanism `entities` already uses
      subject: the first entity, exactly as named in the document
      relation: short snake_case label for the relationship
      object: the second entity, exactly as named in the document
      description: one sentence stating the relationship as the document states it
      strength: {type: number, description: "0-1, how central this relation is"}
```

You will not normally write this by hand — declaring `graph:` injects it
automatically — but it's available for a profile that wants its own shape
instead (a `relations` field the profile defines itself wins over the
auto-injected one).

## 2. Consolidation: `doccat graph build`

Per-document extraction uses *document-local* names — "ACME Corp" exactly as
one document spells it, with no awareness that another document calls the
same company "Acme Corporation." `doccat graph build` reads every record's
`entities`/`relations` and resolves them into **one graph across the corpus**:

```bash
doccat graph build --catalog catalog.jsonl
```

```
Resolved 8,451 document(s) into 2,103 node(s), 3,890 edge(s), 94 communities
Wrote graph (2103 nodes, 3890 edges, 94 communities)
```

```bash
doccat graph stats --catalog catalog.jsonl
```

Options: `--out DIR` (default: `graph/` next to the catalog — same default the
`graph.out_dir` config key uses), `--no-communities`, `--min-community-size N`
(default 2 — smaller clusters aren't reported), `--seed N`, and `--summarize`
(see below). A `doccat.yaml` `graph:` section sets all of these without flags
— distinct from a *profile's* `graph:` block, which controls extraction, not
consolidation:

```yaml
graph:
  out_dir: ./graph
  detect_communities: true
  min_community_size: 2
  summarize: false
  community_seed: 0
```

### Entity resolution

Tier 1 only: case-fold, strip punctuation and common legal-entity suffixes
(`Corp`, `S.L.`, `Ltd.`, …), collapse whitespace — "ACME Corp.", "Acme
Corporation" and "ACME CORP" all resolve to one node; the other spellings are
kept as `aliases`. Deterministic and free, and the entities this corpus
already names formally (companies, registered parties) make it catch the
large majority of real duplicates.

A fuzzy, embedding-based tier 2 — one embedding per *distinct entity name*,
clustered above a similarity threshold — is a documented future upgrade, not
built: tier 1 covers the realistic case, and it is exactly the kind of
"summary-level embedding" hybrid upgrade `DESIGN.md` already flags as
available without changing the architecture, should it ever be needed.

A relation whose subject and object resolve to the *same* node is dropped —
almost always a resolution artifact (two spellings colliding), not a genuine
self-relation.

### Communities

GraphRAG's signature move: detecting clusters of densely-interrelated entities
and summarizing each one. Runs on `networkx`'s built-in Louvain implementation
— pure Python, no extra system dependency. `--summarize` adds one LLM call per
community (far fewer than documents, so cheap in aggregate) generating a short
paragraph of what connects its members, strictly from the relations recorded
— never inventing a connection the graph doesn't show.

### Storage

Plain JSONL, not a graph database — greppable and diffable like
`catalog.jsonl` itself:

```
graph/
  nodes.jsonl        # one resolved entity per line: id, name, aliases, type, doc_ids, degree
  edges.jsonl        # one resolved relation per line: source, target, relation, description, doc_ids, weight
  communities.jsonl  # one community per line: id, members, size, summary
```

A `networkx.MultiDiGraph` is rebuilt from these files on load, never persisted
directly — the on-disk format stays stable even if the in-memory library
choice ever changes. Writes are atomic; a corrupt line is skipped with a
warning on load, same as `catalog.jsonl`.

## 3. Querying: the MCP server's graph tools

| Tool | Purpose |
|---|---|
| `graph_overview` | Orientation — node/edge/community counts, entity types, top relation labels |
| `graph_search` | Fuzzy entity lookup by name or alias |
| `graph_neighbors` | Everything one entity connects to within N hops, direction-agnostic |
| `graph_path` | The shortest chain of stated relations between two entities, or an explicit "not connected" |
| `graph_communities` | List detected clusters (with summaries, if built with `--summarize`), or look up one entity's cluster |

Entity names in `graph_neighbors`/`graph_path`/`graph_communities` are
fuzzy-resolved (`graph_search`'s own matching), so an agent can pass a name
exactly as it appeared in a document without knowing the graph's internal,
normalized id. A catalog with no graph built yet gets a clear hint ("run
`doccat graph build`") from every graph tool rather than an error;
`catalog_overview` reports `has_graph` so an agent knows to try them at all.
The graph hot-reloads the same way the catalog does — rebuild it while the
server is running and the next query picks it up, no restart.

`GRAPH_DIR` (env var, `--graph-dir` on `doccat serve`, or the `graph.out_dir`
config key) locates the directory; default is `graph/` next to the catalog,
matching what `doccat graph build` already writes to.

```
Agent: "How are ACME and the city council connected?"
  → graph_search("ACME") / graph_search("city council") to resolve names
  → graph_path(source="ACME", target="City Council")
  → read_document on the doc_ids backing each hop, to verify against the source
```

## What doesn't change

No existing profile, catalog, prompt, or MCP tool changes behavior for a
profile that doesn't declare `graph:`. The indexing pipeline's parallelism,
journaling and checkpointing are untouched — relations are just a bigger JSON
schema riding the same single call unless `dedicated_pass` is set.
`networkx` is the one new runtime dependency, pure Python, loaded only by
`doccat graph build` and the server's graph tools — a catalog with no `graph:`
profile never imports it.

## Status

Implemented: relation extraction (inline and dedicated-pass), `doccat graph
build` with tier-1 resolution, community detection and optional LLM
summaries, all five MCP tools, and `doccat optimize` proposing the graph
vocabulary. Not built, documented above as a future option: embedding-based
fuzzy entity resolution (tier 2).
