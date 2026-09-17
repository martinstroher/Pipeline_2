"""Fail-fast model configuration tests; no credentials or network are needed."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from src.utils import llm_client

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("LLM_EXTRACTION_MODEL", "LLM_GENERATION_MODEL")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for name in (*MODELS, "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    monkeypatch.setenv("LLM_SEED", "42")
    monkeypatch.setenv(
        "ONTOLOGY_CONFIG_PATH", str(ROOT / "domains/presalt/ontology_config.yaml")
    )


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_required_model_rejects_unset_or_blank(monkeypatch, name, value):
    if value is not None:
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        llm_client.require_model(name)


def test_validation_reports_all_missing_settings():
    with pytest.raises(RuntimeError) as error:
        llm_client.validate_model_settings(*MODELS)
    assert all(name in str(error.value) for name in MODELS)


def test_configured_deployment_is_not_replaced(monkeypatch):
    monkeypatch.setenv("LLM_GENERATION_MODEL", "  custom-deployment  ")
    assert llm_client.require_model("LLM_GENERATION_MODEL") == "custom-deployment"


@pytest.mark.parametrize("explicit", [None, "", " \t"])
def test_generate_rejects_missing_model_before_creating_client(monkeypatch, explicit):
    get_client = Mock(side_effect=AssertionError("Client must not be created"))
    monkeypatch.setattr(llm_client, "get_client", get_client)
    if explicit is not None:
        monkeypatch.setenv("LLM_GENERATION_MODEL", "must-not-be-a-fallback")
    with pytest.raises(RuntimeError, match="model|MODEL"):
        llm_client.generate("test", model=explicit)
    get_client.assert_not_called()


@pytest.mark.parametrize("explicit", [None, "explicit-deployment"])
def test_generate_uses_configured_or_explicit_model(monkeypatch, explicit):
    if explicit is None:
        monkeypatch.setenv("LLM_GENERATION_MODEL", "configured-deployment")
    response = Mock(usage=None)
    response.choices = [Mock(message=Mock(content="ok"), finish_reason="stop")]
    client = Mock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(llm_client, "get_client", lambda: client)
    assert llm_client.generate("test", model=explicit) == "ok"
    assert client.chat.completions.create.call_args.kwargs["model"] == (
        explicit or "configured-deployment"
    )


@pytest.mark.parametrize(("argv", "expected"), [
    ([], MODELS),
    (["--fresh"], MODELS),
    (["--stop-after", "extract"], ("LLM_EXTRACTION_MODEL",)),
    (["--stop-after", "1"], ("LLM_EXTRACTION_MODEL",)),
    (["--stop-after", "2"], ("LLM_EXTRACTION_MODEL",)),
    (["--stop-after", "3"], ("LLM_EXTRACTION_MODEL",)),
    (["--stop-after", "define"], MODELS),
    (["--stop-after", "classify"], MODELS),
    (["--stop-after", "construct"], MODELS),
    (["--stop-after", "validate"], MODELS),
    (["--stop-after", "emit"], MODELS),
    (["--skip-extraction"], ("LLM_GENERATION_MODEL",)),
    (["--skip-extraction", "--stop-after", "extract"], ("LLM_GENERATION_MODEL",)),
    (["--skip-pdf", "--stop-after", "0"], MODELS),
    (["--stop-after", "0"], ()),
    (["--stop-after", "R"], ()),
    (["--taxonomy", "input.csv"], ("LLM_GENERATION_MODEL",)),
    (["--validate", "input.csv", "--validate-emit"], ("LLM_GENERATION_MODEL",)),
    (["--relations", "input.csv"], ("LLM_GENERATION_MODEL",)),
    (["--owl", "input.csv"], ()),
    (["--verify", "input.ttl"], ()),
    (["--taxonomy", "input.csv", "--verify", "input.ttl"], ("LLM_GENERATION_MODEL",)),
    (["--owl", "input.csv", "--relations", "input.csv"], ()),
])
def test_startup_requires_only_reachable_models(argv, expected):
    import pipeline

    args = pipeline._build_parser().parse_args(argv)
    assert pipeline._required_model_settings(args) == expected


@pytest.mark.parametrize(("argv", "model"), [
    (["--stop-after", "extract"], "LLM_EXTRACTION_MODEL"),
    (["--skip-extraction"], "LLM_GENERATION_MODEL"),
    (["--taxonomy", "input.csv"], "LLM_GENERATION_MODEL"),
])
def test_unused_model_settings_do_not_block_startup(monkeypatch, argv, model):
    import pipeline

    monkeypatch.setenv(model, "configured-deployment")
    parser = pipeline._build_parser()
    pipeline._validate_startup(parser.parse_args(argv), parser)


@pytest.mark.parametrize("argv", [
    [], ["--fresh"], ["--stop-after", "extract"],
    ["--taxonomy", "missing.csv"], ["--validate", "missing.csv"],
    ["--relations", "missing.csv"],
])
def test_missing_models_stop_before_dispatch_or_cleanup(monkeypatch, capsys, argv):
    import pipeline

    dispatch = Mock(side_effect=AssertionError("Dispatch must not run"))
    cleanup = Mock(side_effect=AssertionError("Cleanup must not run"))
    monkeypatch.setattr(pipeline, "_dispatch_subcommand", dispatch)
    monkeypatch.setattr(pipeline, "_clean_outputs", cleanup)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", *argv])
    with pytest.raises(SystemExit) as error:
        pipeline.main()
    assert error.value.code == 2
    assert "Missing model configuration" in capsys.readouterr().err
    dispatch.assert_not_called()
    cleanup.assert_not_called()


@pytest.mark.parametrize("argv", [["--validate-emit"], ["--validate-relations", "x.csv"]])
def test_invalid_cli_combinations_keep_their_error(monkeypatch, capsys, argv):
    import pipeline

    monkeypatch.setattr(sys, "argv", ["pipeline.py", *argv])
    with pytest.raises(SystemExit):
        pipeline.main()
    assert "requires --validate" in capsys.readouterr().err


@pytest.mark.parametrize(("module_name", "function", "model"), [
    ("extract.term_extractor", "run_llm_term_extraction", "LLM_EXTRACTION_MODEL"),
    ("define.nld_generator", "run_nld_generation", "LLM_GENERATION_MODEL"),
    ("classify.category_assigner", "run_term_categorization", "LLM_GENERATION_MODEL"),
    ("classify.cq_scorer", "run_cq_refinement", "LLM_GENERATION_MODEL"),
    ("construct.taxonomy_builder", "run_taxonomy_builder", "LLM_GENERATION_MODEL"),
    ("construct.relation_extractor", "run_relation_extraction", "LLM_GENERATION_MODEL"),
    ("validate.critic", "run_critic", "LLM_GENERATION_MODEL"),
])
def test_direct_stage_entry_points_reject_missing_models(tmp_path, module_name, function, model):
    module = importlib.import_module(f"src.modules.{module_name}")
    args = {
        "run_taxonomy_builder": (str(tmp_path / "missing.csv"),),
        "run_critic": (str(tmp_path / "missing.csv"), str(tmp_path / "output")),
    }.get(function, ())
    with pytest.raises(RuntimeError, match=model):
        getattr(module, function)(*args)
    assert not (tmp_path / "output").exists()


def test_live_ablation_does_not_supply_its_own_fallback(monkeypatch, tmp_path):
    from evaluation_study import ablation_study

    monkeypatch.setattr(ablation_study, "OUTPUT_DIR", str(tmp_path / "ablation"))
    get_client = Mock(side_effect=AssertionError("Client must not be created"))
    monkeypatch.setattr(ablation_study, "get_client", get_client)
    with pytest.raises(RuntimeError, match="LLM_GENERATION_MODEL"):
        ablation_study.run_ablation(["B"])
    get_client.assert_not_called()
    assert not (tmp_path / "ablation").exists()


@pytest.mark.parametrize("model", [None, "configured-deployment"])
def test_frozen_only_ablation_stays_offline(monkeypatch, tmp_path, model):
    import pandas as pd

    from evaluation_study import ablation_study
    from src.utils.csv_io import write_csv

    if model is not None:
        monkeypatch.setenv("LLM_GENERATION_MODEL", model)
    terms = tmp_path / "terms.csv"
    nlds = tmp_path / "nlds.csv"
    categories = tmp_path / "categories.csv"
    output = tmp_path / "ablation"
    write_csv(pd.DataFrame([{"Readable_Term": "sample", "Frequency": 1}]), terms)
    write_csv(pd.DataFrame([{
        "Term": "sample", "NLD": "A sample is a material entity.",
        "Context_Used": True, "Context": "Local test evidence.",
    }]), nlds)
    write_csv(pd.DataFrame([{
        "Term": "sample", "NLD": "A sample is a material entity.",
        "Category": "material entity", "Reasoning": "Test classification.",
        "Context_Used": True,
    }]), categories)
    monkeypatch.setenv("FILTERED_TERMS_OUTPUT", str(terms))
    monkeypatch.setenv("ABLATION_FROZEN_A_NLD", str(nlds))
    monkeypatch.setenv("ABLATION_FROZEN_A_CATEGORY", str(categories))
    monkeypatch.setenv("ABLATION_EXPECTED_TERM_COUNT", "1")
    monkeypatch.setattr(ablation_study, "OUTPUT_DIR", str(output))
    get_client = Mock(side_effect=AssertionError("Frozen copying must not create a client"))
    monkeypatch.setattr(ablation_study, "get_client", get_client)
    ablation_study.run_ablation(["A"])
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["model"] == model
    assert (output / "nld_A.csv").read_bytes() == nlds.read_bytes()
    assert (output / "cat_A.csv").read_bytes() == categories.read_bytes()
    get_client.assert_not_called()


def _run_cli(tmp_path, *args):
    bootstrap = """
import runpy, socket, sys
from pathlib import Path
from unittest.mock import patch
script = sys.argv.pop(1)
sys.path.insert(0, str(Path(script).parent))
def deny(*args, **kwargs):
    raise AssertionError("Network access is forbidden in configuration tests")
with patch("dotenv.load_dotenv", return_value=False), \\
     patch.object(socket.socket, "connect", deny), \\
     patch.object(socket.socket, "connect_ex", deny):
    runpy.run_path(script, run_name="__main__")
"""
    env = os.environ.copy()
    for name in (*MODELS, "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                 "LLM_OUTPUT_FILE", "AGGREGATOR_OUTPUT_FILE"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-c", bootstrap, str(ROOT / "pipeline.py"), *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )


def test_real_cli_fails_before_eager_runtime_imports(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    document = inputs / "keep.md"
    document.write_text("Keep this input.")
    result = _run_cli(tmp_path, "--fresh")
    assert result.returncode == 2
    assert all(name in result.stderr for name in MODELS)
    assert "KeyError" not in result.stderr
    assert document.read_text() == "Keep this input."
    assert not (tmp_path / "output").exists()


def test_real_cli_offline_verification_needs_no_model_settings(tmp_path):
    ontology = tmp_path / "sample.ttl"
    ontology.write_text(
        '@prefix owl: <http://www.w3.org/2002/07/owl#> .\n'
        '<https://example.invalid/test> a owl:Ontology .\n'
    )
    result = _run_cli(tmp_path, "--verify", str(ontology), "--skip-oops", "--skip-reasoner")
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "emit_verification.json").read_text())
    assert report["layers"]["syntax"]["status"] == "PASS"


def test_real_cli_export_needs_no_model_settings(tmp_path):
    import pandas as pd

    from rdflib import Graph
    from src.utils.csv_io import write_csv

    taxonomy = tmp_path / "construct_taxonomy.csv"
    write_csv(pd.DataFrame([{
        "Term": "sample", "Parent_Term": "material entity",
        "Category": "material entity", "Relationship_Type": "rdfs:subClassOf",
        "Is_Intermediate": False, "FALLBACK": False,
        "NLD": "A sample is a material entity.",
    }]), taxonomy)
    result = _run_cli(tmp_path, "--owl", str(taxonomy))
    assert result.returncode == 0, result.stderr
    assert len(Graph().parse(tmp_path / "emit_ontology.ttl", format="turtle")) > 0


def test_real_cli_help_needs_no_model_settings(tmp_path):
    result = _run_cli(tmp_path, "--help")
    assert result.returncode == 0
    assert "--stop-after" in result.stdout
