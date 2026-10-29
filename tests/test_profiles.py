"""Profile loading, inheritance, routing and excerpting."""

from __future__ import annotations

import pytest

from doccat.profiles import (BASE_PROFILE, MatchContext, ProfileError,
                             load_profiles)


@pytest.fixture(scope="module")
def profiles():
    return load_profiles()


def test_builtin_profiles_load(profiles):
    assert BASE_PROFILE in profiles
    assert {"legal", "parliamentary", "technical", "financial", "academic",
            "intranet", "minutes", "long_document"} <= set(profiles.names())


def test_inheritance_keeps_core_schema_and_adds_domain_fields(profiles):
    generic, legal = profiles.get("generic"), profiles.get("legal")
    assert set(generic.fields) <= set(legal.fields)
    assert "parties" in legal.fields and "parties" not in generic.fields
    # the child's own vocabulary replaces the parent's
    assert "framework_agreement" in legal.doc_types
    assert "invoice" not in legal.doc_types


def test_match_rules_are_not_inherited(profiles):
    # legal declares match rules; a profile extending it without `match`
    # must not silently claim legal's documents.
    assert not profiles.get("legal").match.empty
    assert profiles.get("generic").match.empty


def test_routing_prefers_the_highest_priority_match(profiles):
    ctx = MatchContext(path="contracts/supply_contract_2024.pdf", filetype="pdf",
                       size_bytes=10_000)
    assert profiles.select(ctx).name == "legal"

    ctx = MatchContext(path="manuals/installation_guide.pdf", filetype="pdf", size_bytes=5_000)
    assert profiles.select(ctx).name == "technical"

    ctx = MatchContext(path="random/whatever.pdf", filetype="pdf", size_bytes=5_000)
    assert profiles.select(ctx).name == "generic"


def test_size_rule_routes_huge_files_to_map_reduce(profiles):
    ctx = MatchContext(path="random/huge.pdf", filetype="pdf", size_bytes=50_000_000)
    assert profiles.select(ctx).name == "long_document"


def test_json_record_metadata_routing(tmp_path):
    (tmp_path / "senado.yaml").write_text(
        "name: senado\nextends: parliamentary\npriority: 60\n"
        "match:\n  metadata_equals:\n    orga: Senado\n", encoding="utf-8")
    profiles = load_profiles(extra_dirs=[tmp_path])
    ctx = MatchContext(path="Plenos.json#123", source_kind="json",
                       filetype="json_record", metadata={"orga": "Senado"})
    chosen = profiles.select(ctx)
    assert chosen.name == "senado"
    # inherited the parliamentary schema
    assert "chamber" in chosen.fields and "agenda_items" in chosen.fields


def test_prompt_contains_schema_vocabulary_and_guidance(profiles):
    prompt = profiles.get("financial").build_prompt("invoice_42.pdf", excerpt="TOTAL 120 EUR")
    assert "financial records cataloguer" in prompt
    assert '"total_amount"' in prompt
    assert "credit_note" in prompt          # doc_types substituted
    assert "{doc_types}" not in prompt
    assert "TOTAL 120 EUR" in prompt


def test_excerpt_strategies():
    profiles = load_profiles()
    text = "".join(f"{i:06d}" for i in range(20_000))  # 120k chars

    head_tail = profiles.get("generic")
    ex = head_tail.excerpt(text)
    assert ex.startswith(text[:100]) and ex.endswith(text[-100:])
    assert "middle of document omitted" in ex

    sampled = profiles.get("parliamentary")
    ex = sampled.excerpt(text)
    assert ex.count("jumping to offset") == sampled.samples
    assert len(ex) <= sampled.max_chars + 500

    full = profiles.get("minutes")
    assert full.excerpt(text) == text[: full.max_chars]


def test_map_reduce_chunking_is_bounded():
    profile = load_profiles().get("long_document")
    text = "x" * 10_000_000
    chunks = profile.chunks(text)
    assert len(chunks) == profile.max_chunks
    assert chunks[-1].endswith("[...document continues beyond the indexed portion...]")
    assert profile.wants_map_reduce(text)
    assert not profile.wants_map_reduce("short")


def test_normalize_coerces_model_output(profiles):
    legal = profiles.get("legal")
    out = legal.normalize({"keywords": "a, b, c", "parties": "ACME",
                           "entities": {"people": "Ada"}, "title": 42})
    assert out["keywords"] == ["a", "b", "c"]
    assert out["parties"] == ["ACME"]
    assert out["entities"]["people"] == ["Ada"]
    assert out["title"] == "42"


def test_missing_core_fields_are_reported(profiles):
    assert profiles.get("generic").missing_core_fields({"title": "x"})


def test_unknown_keys_are_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: bad\nnonsense: true\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="unknown top-level key"):
        load_profiles(extra_dirs=[tmp_path])


def test_bad_strategy_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "name: bad\nexcerpt:\n  strategy: telepathy\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="unknown excerpt strategy"):
        load_profiles(extra_dirs=[tmp_path])


def test_circular_inheritance_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("name: a\nextends: b\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("name: b\nextends: a\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="circular"):
        load_profiles(extra_dirs=[tmp_path])


def test_user_profile_can_override_a_builtin(tmp_path):
    (tmp_path / "legal.yaml").write_text(
        "name: legal\ndescription: House rules\nextends: generic\n"
        "prompt:\n  role: Custom legal role\n", encoding="utf-8")
    profiles = load_profiles(extra_dirs=[tmp_path])
    assert profiles.get("legal").role == "Custom legal role"
    assert "parties" not in profiles.get("legal").fields


# ---------------------------------------------------------------------------
# Graph extraction: list-of-objects fields, the `graph:` block, prompts
# ---------------------------------------------------------------------------

def test_fieldspec_list_of_objects_placeholder_and_coerce():
    from doccat.profiles import FieldSpec

    spec = FieldSpec.parse("relations", {
        "type": "list", "items": "object",
        "fields": {
            "subject": "first entity",
            "relation": "short label",
            "object": "second entity",
            "strength": {"type": "number", "description": "0-1"},
        },
    })
    assert spec.placeholder() == [
        {"subject": "first entity", "relation": "short label",
         "object": "second entity", "strength": "0-1"}
    ]
    coerced = spec.coerce([
        {"subject": "ACME", "relation": "supplies_to", "object": "Contoso", "strength": "0.9"},
        "a bare string, not an object",
        {"subject": "X"},
    ])
    # the bare string is dropped; everything else is kept, numbers coerced
    assert coerced == [
        {"subject": "ACME", "relation": "supplies_to", "object": "Contoso", "strength": 0.9},
        {"subject": "X"},
    ]


def test_fieldspec_plain_list_is_unaffected_by_items_object():
    from doccat.profiles import FieldSpec

    plain = FieldSpec.parse("keywords", {"type": "list", "description": "kw"})
    assert plain.items == "string"
    assert plain.placeholder() == ["kw"]
    assert plain.coerce("a, b, c") == ["a", "b", "c"]
    assert plain.coerce(None) is None


@pytest.mark.parametrize("bad_spec,match", [
    ({"type": "list", "items": "object", "fields": {}}, "non-empty 'fields'"),
    ({"type": "string", "items": "object"}, "only applies to type: list"),
    ({"type": "list", "items": "bogus"}, "must be 'string' or 'object'"),
])
def test_fieldspec_rejects_invalid_items_configurations(bad_spec, match):
    from doccat.profiles import FieldSpec

    with pytest.raises(ProfileError, match=match):
        FieldSpec.parse("x", bad_spec)


def test_graph_block_enables_relations_field_and_is_absent_by_default(profiles):
    generic = profiles.get("generic")
    assert not generic.graph_enabled
    assert "relations" not in generic.fields

    legal = profiles.get("legal")
    assert legal.graph_enabled
    assert "relations" in legal.fields
    assert legal.fields["relations"].type == "list"
    assert legal.fields["relations"].items == "object"
    assert set(legal.fields["relations"].fields) == {
        "subject", "relation", "object", "description", "strength"}


def test_graph_vocabulary_is_inherited_like_excerpt_and_llm_settings(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "name: base\nextends: generic\ngraph:\n  entity_types: [organization]\n"
        "  relation_types: [relates_to]\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "name: child\nextends: base\n", encoding="utf-8")  # no graph: block of its own
    profiles = load_profiles(extra_dirs=[tmp_path])

    child = profiles.get("child")
    assert child.graph_enabled
    assert child.graph.relation_types == ["relates_to"]
    assert "relations" in child.fields


def test_graph_block_rejects_unknown_keys_and_bad_types(tmp_path):
    (tmp_path / "bad1.yaml").write_text(
        "name: bad1\ngraph:\n  nonsense: true\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="unknown graph key"):
        load_profiles(extra_dirs=[tmp_path])

    (tmp_path / "bad1.yaml").unlink()
    (tmp_path / "bad2.yaml").write_text(
        "name: bad2\ngraph:\n  min_confidence: not-a-number\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="min_confidence must be a number"):
        load_profiles(extra_dirs=[tmp_path])


def test_a_profile_can_write_its_own_relations_field_instead(tmp_path):
    (tmp_path / "custom.yaml").write_text("""
name: custom
extends: generic
graph:
  entity_types: [thing]
prompt:
  fields:
    relations:
      type: list
      description: a custom, non-standard relations shape
""", encoding="utf-8")
    profiles = load_profiles(extra_dirs=[tmp_path])
    custom = profiles.get("custom")
    assert custom.graph_enabled
    # the profile's own scalar-list field wins over the auto-injected object schema
    assert custom.fields["relations"].items == "string"


def test_build_prompt_includes_relation_vocabulary_as_a_hint(profiles):
    prompt = profiles.get("legal").build_prompt("x.pdf", excerpt="ACME supplies Contoso.")
    assert "supplies_to" in prompt
    assert '"relations"' in prompt
    assert "hint, not a constraint" in prompt


def test_build_prompt_can_exclude_relations_for_a_dedicated_pass(profiles):
    legal = profiles.get("legal")
    full = legal.build_prompt("x.pdf")
    without_relations = legal.build_prompt("x.pdf", exclude_fields=frozenset({"relations"}))
    assert '"relations"' in full
    assert '"relations"' not in without_relations
    # entity vocabulary still applies (it still helps the `entities` field);
    # relation vocabulary does not, since relations aren't being asked for here
    assert "organization" in without_relations or "jurisdiction" in without_relations
    assert "supplies_to" not in without_relations


def test_build_relations_prompt_grounds_in_known_entities(profiles):
    legal = profiles.get("legal")
    prompt = legal.build_relations_prompt(
        "x.pdf", "ACME supplies Contoso.",
        {"organizations": ["ACME", "Contoso"], "people": []})
    assert "ACME" in prompt and "Contoso" in prompt
    assert '"relations"' in prompt
    assert "supplies_to" in prompt       # relation_types still guide the dedicated call


def test_build_relations_prompt_without_a_relations_field_raises():
    generic = load_profiles().get("generic")
    with pytest.raises(ValueError, match="relations"):
        generic.build_relations_prompt("x.pdf", "text", {})


def test_filter_relations_drops_only_below_threshold(tmp_path):
    (tmp_path / "strict.yaml").write_text(
        "name: strict\nextends: generic\ngraph:\n  min_confidence: 0.5\n", encoding="utf-8")
    profile = load_profiles(extra_dirs=[tmp_path]).get("strict")
    record = {"relations": [
        {"subject": "A", "strength": 0.9},
        {"subject": "B", "strength": 0.1},
        {"subject": "C"},                 # no strength at all — kept, never penalised
    ]}
    dropped = profile.filter_relations(record)
    assert dropped == 1
    assert record["relations"] == [{"subject": "A", "strength": 0.9}, {"subject": "C"}]


def test_filter_relations_is_a_noop_without_graph_or_threshold(profiles):
    generic = profiles.get("generic")
    record = {"relations": [{"subject": "A", "strength": 0.0}]}
    assert generic.filter_relations(record) == 0        # no graph config at all

    legal = profiles.get("legal")                        # graph enabled, min_confidence=0
    record2 = {"relations": [{"subject": "A", "strength": 0.0}]}
    assert legal.filter_relations(record2) == 0
    assert record2["relations"] == [{"subject": "A", "strength": 0.0}]


def test_summary_line_reports_graph_extraction(profiles):
    assert "graph extraction" in profiles.get("legal").summary_line()
    assert "dedicated pass" in profiles.get("parliamentary").summary_line()
    assert "graph" not in profiles.get("generic").summary_line()
