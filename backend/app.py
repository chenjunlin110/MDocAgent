from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
MDOC_PATH = ROOT / "MDocAgent"
CONFIG_DIR = MDOC_PATH / "config"
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DATA_DIR = ROOT / "data" / "upload-datasets"
UPLOAD_TMP_DIR = ROOT / "tmp" / "uploads"
UPLOAD_MANIFEST = UPLOAD_DIR / "manifest.json"
SETTINGS_PATH = ROOT / "data" / "settings.json"
MODEL_LIST_PATH = ROOT / "data" / "models.json"
DEFAULT_UPLOAD_QUESTION = "Summarize this document."

sys.path.append(str(MDOC_PATH))

from mydatasets.base_dataset import BaseDataset  # noqa: E402
from agents.mdoc_agent import MDocAgent  # noqa: E402
from retrieval.image_retrieval import ColpaliRetrieval  # noqa: E402


class QARequest(BaseModel):
    upload_id: str = Field(..., description="Upload id")
    question: str = Field(..., description="User question")
    top_k: int = Field(4, ge=1, le=50, description="Top-k pages used during inference")
    use_openai: bool = Field(False, description="Use OpenAI API for all agents")


class AgentTurn(BaseModel):
    role: str
    text: str


class QAResponse(BaseModel):
    answer: str
    agents: List[AgentTurn]
    critics: dict


class SettingsPayload(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


app = FastAPI(title="MDocAgent API")


@app.on_event("startup")
def _warm_retrieval_models() -> None:
    # Preload the image retrieval model once at startup to avoid first-request latency.
    try:
        cfg = _compose_cfg(["retrieval=image"], use_openai=False)
        ColpaliRetrieval(cfg.retrieval)
    except Exception as exc:
        print(f"Warning: failed to preload retrieval model: {exc}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_filename(name: str) -> str:
    cleaned = Path(name).name.strip().replace(" ", "_")
    return cleaned or "upload.pdf"


def _load_manifest() -> List[dict]:
    if not UPLOAD_MANIFEST.exists():
        return []
    try:
        with UPLOAD_MANIFEST.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    return []


def _write_manifest(items: List[dict]) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with UPLOAD_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _write_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def _merge_settings(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    return merged


def _load_model_list() -> List[dict]:
    if not MODEL_LIST_PATH.exists():
        raise HTTPException(status_code=400, detail="models.json not found")
    try:
        with MODEL_LIST_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"models.json is invalid: {exc}") from exc
    if isinstance(data, list):
        items = []
        for entry in data:
            if isinstance(entry, str):
                items.append({"id": entry})
            elif isinstance(entry, dict) and entry.get("id"):
                items.append({"id": str(entry["id"])})
        return items
    raise HTTPException(status_code=400, detail="models.json must be a list")

def _upsert_manifest(record: dict) -> dict:
    items = _load_manifest()
    for index, item in enumerate(items):
        if item.get("id") == record.get("id"):
            items[index] = record
            _write_manifest(items)
            return record
    items.append(record)
    _write_manifest(items)
    return record


def _get_upload_record(upload_id: str) -> dict | None:
    for item in _load_manifest():
        if item.get("id") == upload_id:
            return item
    return None


def _remove_upload_record(upload_id: str) -> None:
    items = [item for item in _load_manifest() if item.get("id") != upload_id]
    _write_manifest(items)


def _upload_dataset_info(upload_id: str, stored_name: str) -> dict:
    dataset_name = f"upload-{upload_id}"
    data_dir = UPLOAD_DATA_DIR / upload_id
    extract_dir = UPLOAD_TMP_DIR / upload_id
    sample_path = data_dir / "samples.json"
    sample_with_retrieval_path = data_dir / "sample-with-retrieval-results.json"
    result_dir = ROOT / "results" / dataset_name
    overrides = [
        "+dataset=base",
        f"dataset.name={dataset_name}",
        f"dataset.data_dir={data_dir.as_posix()}",
        f"dataset.document_path={UPLOAD_DIR.as_posix()}",
        f"dataset.extract_path={extract_dir.as_posix()}",
        f"dataset.sample_path={sample_path.as_posix()}",
        f"dataset.sample_with_retrieval_path={sample_with_retrieval_path.as_posix()}",
        f"dataset.result_dir={result_dir.as_posix()}",
    ]
    return {
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "extract_dir": extract_dir,
        "sample_path": sample_path,
        "sample_with_retrieval_path": sample_with_retrieval_path,
        "result_dir": result_dir,
        "overrides": overrides,
        "doc_id": stored_name,
    }


def _delete_upload_assets(record: dict) -> None:
    stored_name = record.get("stored_name", "")
    if stored_name:
        upload_path = UPLOAD_DIR / stored_name
        if upload_path.exists():
            upload_path.unlink()

    info = _upload_dataset_info(record["id"], stored_name or "")
    for path in (
        info["data_dir"],
        info["extract_dir"],
        info["result_dir"],
    ):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _ensure_sample_file(sample_path: Path, doc_id: str, question: str) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample = {"doc_id": doc_id, "question": question}
    if sample_path.exists():
        try:
            with sample_path.open("r", encoding="utf-8") as handle:
                samples = json.load(handle)
        except json.JSONDecodeError:
            samples = []
        if samples:
            samples[0] = {**samples[0], **sample}
        else:
            samples = [sample]
    else:
        samples = [sample]
    with sample_path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=4)


def _parse_critical_info(text: str) -> dict:
    start_index = text.find("{")
    end_index = text.find("}") + 1
    if start_index == -1 or end_index <= 0:
        return {"text": "", "image": ""}
    payload = text[start_index:end_index]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"text": "", "image": ""}


def _render_context(turns: List[AgentTurn]) -> str:
    if not turns:
        return ""
    lines = ["Context from previous agents:"]
    for turn in turns:
        lines.append(f"{turn.role} Agent:\n{turn.text}")
    return "\n".join(lines)


def _build_question(question: str, context: str, clue: str) -> str:
    parts = []
    if context:
        parts.append(context)
    if clue:
        parts.append(f"Clue:\n{clue}")
    parts.append(f"Question:\n{question}")
    return "\n\n".join(parts)


def _apply_openai_override(agent_cfg, model_name: str, api_key: str, base_url: str):
    agent_cfg.model = compose(config_name="model/openai").model
    agent_cfg.model.model = model_name
    agent_cfg.model.api_key = api_key
    agent_cfg.model.base_url = base_url


def _compose_cfg(overrides: List[str], use_openai: bool):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        cfg = compose(config_name="base", overrides=overrides)
        settings = _load_settings()
        custom_model = settings.get("model") or ""
        custom_api_key = settings.get("api_key") or ""
        custom_base_url = settings.get("base_url") or ""
        use_custom = bool(custom_model and custom_api_key and custom_base_url)
        for agent_cfg in cfg.mdoc_agent.agents:
            agent_cfg.agent = compose(config_name=f"agent/{agent_cfg.agent}").agent
            if use_custom:
                _apply_openai_override(agent_cfg, custom_model, custom_api_key, custom_base_url)
            elif use_openai:
                agent_cfg.model = compose(config_name="model/openai").model
            else:
                agent_cfg.model = compose(config_name=f"model/{agent_cfg.model}").model
        cfg.mdoc_agent.sum_agent.agent = compose(config_name=f"agent/{cfg.mdoc_agent.sum_agent.agent}").agent
        if use_custom:
            _apply_openai_override(
                cfg.mdoc_agent.sum_agent,
                custom_model,
                custom_api_key,
                custom_base_url,
            )
        elif use_openai:
            cfg.mdoc_agent.sum_agent.model = compose(config_name="model/openai").model
        else:
            cfg.mdoc_agent.sum_agent.model = compose(config_name=f"model/{cfg.mdoc_agent.sum_agent.model}").model

    if use_openai:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        for agent_cfg in cfg.mdoc_agent.agents:
            agent_cfg.model.api_key = agent_cfg.model.api_key or api_key
            agent_cfg.model.base_url = agent_cfg.model.base_url or base_url
        cfg.mdoc_agent.sum_agent.model.api_key = cfg.mdoc_agent.sum_agent.model.api_key or api_key
        cfg.mdoc_agent.sum_agent.model.base_url = cfg.mdoc_agent.sum_agent.model.base_url or base_url

    return cfg


def _update_sample_for_question(dataset: BaseDataset, question: str) -> dict:
    samples = dataset.load_data(use_retreival=True)
    if not samples:
        raise HTTPException(status_code=500, detail="No samples found for this upload")
    sample = samples[0]
    sample[dataset.config.question_key] = question
    for key in (
        dataset.config.r_text_key,
        f"{dataset.config.r_text_key}_score",
        dataset.config.r_image_key,
        f"{dataset.config.r_image_key}_score",
    ):
        sample.pop(key, None)
    dataset.dump_data(samples, use_retreival=True)
    return sample


def _run_retrieval(cfg, retrieval_type: str, prepare_only: bool = False) -> None:
    try:
        module_name, class_name = cfg.retrieval.class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        retrieval_class = getattr(module, class_name)
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval dependency missing: {exc.name}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load retrieval: {exc}") from exc

    dataset = BaseDataset(cfg.dataset)
    retrieval_model = retrieval_class(cfg.retrieval)

    if prepare_only:
        retrieval_model.prepare(dataset)
        return

    if retrieval_type == "text":
        retrieval_model.find_top_k(dataset, force_prepare=False)
    else:
        retrieval_model.find_top_k(dataset, prepare=False)


def _ingest_upload(record: dict) -> None:
    info = _upload_dataset_info(record["id"], record["stored_name"])
    _ensure_sample_file(info["sample_path"], info["doc_id"], DEFAULT_UPLOAD_QUESTION)

    base_cfg = _compose_cfg(info["overrides"], use_openai=False)
    dataset = BaseDataset(base_cfg.dataset)
    dataset.extract_content()

    text_cfg = _compose_cfg(info["overrides"] + ["retrieval=text"], use_openai=False)
    _run_retrieval(text_cfg, "text", prepare_only=True)

    image_cfg = _compose_cfg(info["overrides"] + ["retrieval=image"], use_openai=False)
    _run_retrieval(image_cfg, "image", prepare_only=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/settings")
def get_settings():
    settings = _load_settings()
    return {
        "api_key_present": bool(settings.get("api_key")),
        "base_url": settings.get("base_url", ""),
        "model": settings.get("model", ""),
    }


@app.post("/settings")
def update_settings(payload: SettingsPayload):
    updates = payload.model_dump()
    existing = _load_settings()
    merged = _merge_settings(existing, updates)
    model_id = updates.get("model")
    if model_id:
        api_key = merged.get("api_key") or ""
        base_url = merged.get("base_url") or ""
        if not api_key or not base_url:
            raise HTTPException(status_code=400, detail="api_key and base_url are required to set model")
        items = _load_model_list()
        model_ids = [item["id"] for item in items]
        if model_ids and model_id not in model_ids:
            raise HTTPException(status_code=400, detail="Selected model is not in models.json")
    _write_settings(merged)
    return {
        "api_key_present": bool(merged.get("api_key")),
        "base_url": merged.get("base_url", ""),
        "model": merged.get("model", ""),
    }


@app.get("/bedrock/models")
def list_bedrock_models():
    items = _load_model_list()
    models = [{"id": item["id"]} for item in items]
    return {"models": models, "warning": ""}


@app.get("/uploads")
def list_uploads():
    items = _load_manifest()
    return {"uploads": items}


@app.post("/uploads")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(file.filename)
    file_id = uuid.uuid4().hex
    stored_name = f"{file_id}_{safe_name}"
    out_path = UPLOAD_DIR / stored_name
    with out_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    file.file.close()

    record = {
        "id": file_id,
        "filename": safe_name,
        "stored_name": stored_name,
        "path": str(out_path),
        "size": out_path.stat().st_size,
        "status": "processing",
    }
    _upsert_manifest(record)

    try:
        _ingest_upload(record)
        record["status"] = "ready"
        _upsert_manifest(record)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        _upsert_manifest(record)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return record


@app.post("/qa", response_model=QAResponse)
def qa(payload: QARequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    record = _get_upload_record(payload.upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    if record.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Upload is not ready")

    info = _upload_dataset_info(record["id"], record["stored_name"])
    overrides = info["overrides"] + [f"dataset.top_k={payload.top_k}"]

    base_cfg = _compose_cfg(overrides + ["retrieval=text"], use_openai=payload.use_openai)
    dataset = BaseDataset(base_cfg.dataset)

    _update_sample_for_question(dataset, question)

    text_cfg = _compose_cfg(overrides + ["retrieval=text"], use_openai=False)
    _run_retrieval(text_cfg, "text")

    image_cfg = _compose_cfg(overrides + ["retrieval=image"], use_openai=False)
    _run_retrieval(image_cfg, "image")

    samples = dataset.load_data(use_retreival=True)
    if not samples:
        raise HTTPException(status_code=500, detail="No samples found after retrieval")
    sample = samples[0]
    _, texts, images = dataset.load_sample_retrieval_data(sample)

    mdoc_agent = MDocAgent(base_cfg.mdoc_agent)

    conversation: List[AgentTurn] = []
    general_agent = mdoc_agent.agents[-1]
    general_answer, _ = general_agent.predict(question, texts=texts, images=images, with_sys_prompt=True)
    conversation.append(AgentTurn(role="General", text=general_answer))

    critical_info = general_agent.self_reflect(
        prompt=general_agent.config.agent.critical_prompt,
        add_to_message=False,
    )
    critics = _parse_critical_info(critical_info)

    text_agent = mdoc_agent.agents[1]
    text_context = _render_context(conversation)
    text_question = _build_question(question, text_context, critics.get("text", ""))
    text_answer, _ = text_agent.predict(text_question, texts=texts, images=None, with_sys_prompt=True)
    conversation.append(AgentTurn(role="Text", text=text_answer))

    image_agent = mdoc_agent.agents[0]
    image_context = _render_context(conversation)
    image_question = _build_question(question, image_context, critics.get("image", ""))
    image_answer, _ = image_agent.predict(image_question, texts=None, images=images, with_sys_prompt=True)
    conversation.append(AgentTurn(role="Image", text=image_answer))

    summary_input = "Question:\n" + question + "\n\n" + "\n\n".join(
        [f"{turn.role} Agent:\n{turn.text}" for turn in conversation]
    )
    final_answer, _ = mdoc_agent.sum(summary_input)
    mdoc_agent.clean_messages()

    return QAResponse(
        answer=final_answer,
        agents=conversation,
        critics=critics,
    )


@app.delete("/uploads/{upload_id}")
def delete_upload(upload_id: str):
    record = _get_upload_record(upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    _delete_upload_assets(record)
    _remove_upload_record(upload_id)
    return {"status": "deleted", "id": upload_id}
