"""Offline dependency smoke tests; all generated files stay in pytest's tmp_path."""

import json
import os
import socket
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch, tmp_path):
    def deny_connection(*args, **kwargs):
        raise AssertionError("The runtime smoke tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_connection)
    monkeypatch.setattr(socket, "create_connection", deny_connection)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    for key, value in dotenv_values(ROOT / ".env.example").items():
        if value is not None and not key.startswith("AZURE_"):
            monkeypatch.setenv(key, value)
    for key in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setenv("OOPS_URL", "")
    monkeypatch.setenv(
        "ONTOLOGY_CONFIG_PATH", str(ROOT / "domains/presalt/ontology_config.yaml")
    )


def test_cli_import_and_spacy_model():
    import pipeline
    import spacy

    args = pipeline._build_parser().parse_args(["--stop-after", "extract"])
    assert args.stop_after == "extract"
    nlp = spacy.load("en_core_web_sm")
    assert [token.lemma_ for token in nlp("rocks")] == ["rock"]


def test_azure_request_contract_with_mock_transport(monkeypatch, tmp_path):
    import httpx
    from openai import OpenAI

    from src.utils import llm_client

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": '[{"term":"sample"}]'},
                "finish_reason": "stop",
            }],
        })

    with OpenAI(
        api_key="offline-placeholder",
        base_url="https://example.invalid/openai/v1/",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(llm_client, "_client", client)
        monkeypatch.setattr(llm_client, "_USAGE_LOG", str(tmp_path / "usage.csv"))
        result = llm_client.generate(
            "Return sample JSON",
            system_instruction="You are a test expert.",
            response_mime_type="application/json",
            temperature=0.0,
        )
    assert json.loads(result) == [{"term": "sample"}]
    assert len(requests) == 1
    assert requests[0].url.path == "/openai/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-5.4"
    assert body["reasoning_effort"] == "high"
    assert body["seed"] == 42
    assert body["max_completion_tokens"] == 32000
    assert body["messages"][0]["role"] == "system"
    assert "temperature" not in body
    assert "response_format" not in body


def test_pdf_conversion(tmp_path):
    from src.utils.pdf_processor import process_folder
    import pymupdf

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Porosity describes void space in a rock.")
        document.save(tmp_path / "sample.pdf")
    process_folder(str(tmp_path))
    assert "Porosity describes void space" in (tmp_path / "sample.md").read_text()


def test_local_inference_configuration_disables_telemetry(monkeypatch):
    import onnxruntime
    from src.utils.onnx_runtime import configure_onnx_runtime

    disable = Mock()
    monkeypatch.setattr(onnxruntime, "disable_telemetry_events", disable)
    configure_onnx_runtime()
    disable.assert_called_once_with()


@pytest.mark.parametrize("attempt", range(3))
def test_pdf_and_retrieval_runtime_exits_cleanly(tmp_path, attempt):
    script = """
import socket, sys
from pathlib import Path
def deny(*args, **kwargs):
    raise AssertionError("Network forbidden")
socket.socket.connect = deny
socket.socket.connect_ex = deny
from src.utils.pdf_processor import process_folder
from src.utils import rag_setup
import pymupdf
root = Path(sys.argv[1])
with pymupdf.open() as document:
    page = document.new_page()
    page.insert_text((72, 72), "A specimen is selected for examination.")
    document.save(root / "sample.pdf")
process_folder(str(root))
assert "specimen" in (root / "sample.md").read_text()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Attempt {attempt}: {result.stdout}\n{result.stderr}"


def test_retrieval_cache_and_reranking_without_model_downloads(monkeypatch, tmp_path):
    from src.utils import rag_setup

    class LocalEmbeddings:
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            return [float(text.lower().count("porosity")), 1.0]

    class LocalReranker:
        def predict(self, pairs):
            return [float("porosity" in text.lower()) for _, text in pairs]

    monkeypatch.setattr(rag_setup, "CHROMA_DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_setup, "_BM25_RETRIEVER", None)
    monkeypatch.setattr(rag_setup, "get_embedding_model", lambda: LocalEmbeddings())
    monkeypatch.setattr(rag_setup, "get_cross_encoder", lambda: LocalReranker())
    (tmp_path / "sample.md").write_text(
        "# Porosity\nPorosity measures void space.\n# Calcite\nCalcite is a mineral.\n"
    )
    documents = rag_setup.load_documents(str(tmp_path))
    chunks = rag_setup.split_documents(documents)
    assert len(chunks) == 2
    rag_setup.create_vector_store(chunks, chunk_size=1024)
    store = rag_setup.load_vector_store(chunk_size=1024)
    sparse = rag_setup.get_bm25_retriever(chunks)
    results = rag_setup.get_relevant_documents(
        "porosity", store, sparse, search_k=2, rerank_k=1
    )
    assert len(results) == 1
    assert "Porosity" in results[0][0].page_content
    assert results[0][1] == 1.0


def test_retrieval_models_use_pinned_standard_code(monkeypatch):
    from src.utils import rag_setup

    embeddings = Mock()
    reranker = Mock()
    monkeypatch.setattr(rag_setup, "HuggingFaceEmbeddings", embeddings)
    monkeypatch.setattr(rag_setup, "CrossEncoder", reranker)
    monkeypatch.setattr(rag_setup, "_EMBEDDINGS", None)
    monkeypatch.setattr(rag_setup, "_CROSS_ENCODER", None)

    rag_setup.get_embedding_model()
    rag_setup.get_cross_encoder()

    assert embeddings.call_args.kwargs["model_kwargs"] == {
        "device": "cpu",
        "revision": rag_setup.EMBED_MODEL_REVISION,
        "trust_remote_code": False,
    }
    assert reranker.call_args.kwargs == {
        "revision": rag_setup.RERANK_MODEL_REVISION,
        "trust_remote_code": False,
    }


def test_repeatable_export_and_offline_verification(tmp_path):
    import pandas as pd
    from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef
    from rdflib.compare import isomorphic

    from src.modules.emit.owl_exporter import run_owl_export
    from src.modules.emit.verifier import run_ontology_verification
    from src.utils.csv_io import write_csv
    from src.utils.ontology_config import get_config

    taxonomy = tmp_path / "taxonomy.csv"
    write_csv(pd.DataFrame([{
        "Term": "sample material",
        "Parent_Term": "material entity",
        "Relationship_Type": "rdfs:subClassOf",
        "Category": "material entity",
        "Is_Intermediate": False,
        "NLD": "A sample material is a material entity used in a test.",
        "FALLBACK": False,
    }]), taxonomy)
    graphs = []
    for name in ("first", "second"):
        ttl_path = tmp_path / f"{name}.ttl"
        run_owl_export(str(taxonomy), output_path=str(ttl_path))
        graphs.append(Graph().parse(ttl_path, format="turtle"))
    assert isomorphic(*graphs)
    term = URIRef(get_config().project_namespace() + "SampleMaterial")
    assert (term, RDF.type, OWL.Class) in graphs[0]
    assert (term, RDFS.label, Literal("Sample Material", lang="en")) in graphs[0]
    assert (term, RDFS.subClassOf, term) not in graphs[0]
    report = run_ontology_verification(
        str(tmp_path / "first.ttl"),
        output_path=str(tmp_path / "verification.json"),
        skip_oops=True,
        skip_reasoner=True,
    )
    assert report["layers"]["syntax"]["status"] == "PASS"
    assert report["layers"]["oops_pitfalls"]["status"] == "SKIP"
    assert report["layers"]["reasoner"]["status"] == "SKIP"
