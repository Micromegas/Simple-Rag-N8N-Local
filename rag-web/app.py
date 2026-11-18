import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama-metrics:8080").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_web_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
N8N_CHAT_WEBHOOK_URL = os.getenv("N8N_CHAT_WEBHOOK_URL", "").rstrip("/")
N8N_INGEST_WEBHOOK_URL = os.getenv("N8N_INGEST_WEBHOOK_URL", "").rstrip("/")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MAX_FILE_BYTES = 30 * 1024 * 1024
STATIC_FILE = Path(__file__).with_name("index.html")

app = FastAPI(title="RAG Lab", docs_url=None, redoc_url=None)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=200)


def qdrant_request(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{QDRANT_URL}{path}", timeout=30, **kwargs)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"No pude conectar con Qdrant: {exc}") from exc
    if not response.ok:
        raise HTTPException(status_code=502, detail=f"Qdrant respondió {response.status_code}: {response.text[:400]}")
    return response.json() if response.content else {}


def ollama_request(path: str, payload: dict, timeout: int = 900):
    try:
        response = requests.post(f"{OLLAMA_URL}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"No pude conectar con Ollama: {exc}") from exc
    if not response.ok:
        raise HTTPException(status_code=502, detail=f"Ollama respondió {response.status_code}: {response.text[:400]}")
    return response.json()


def embed_texts(texts: list[str]) -> list[list[float]]:
    result = ollama_request("/api/embed", {"model": EMBEDDING_MODEL, "input": texts}, timeout=900)
    vectors = result.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise HTTPException(status_code=502, detail="Ollama no devolvió todos los embeddings esperados.")
    return vectors


def ensure_collection(vector_size: int):
    path = f"/collections/{COLLECTION}"
    try:
        response = requests.get(f"{QDRANT_URL}{path}", timeout=15)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"No pude conectar con Qdrant: {exc}") from exc
    if response.status_code == 200:
        info = response.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
        if isinstance(info, dict) and "size" in info and info["size"] != vector_size:
            raise HTTPException(
                status_code=409,
                detail=f"La colección {COLLECTION} usa vectores de {info['size']} dimensiones; el modelo actual genera {vector_size}. Cambiá la colección o reindexá con un modelo compatible.",
            )
        return
    if response.status_code != 404:
        raise HTTPException(status_code=502, detail=f"No pude consultar la colección de Qdrant: {response.text[:400]}")
    qdrant_request("PUT", path, json={"vectors": {"size": vector_size, "distance": "Cosine"}})


def collection_exists() -> bool:
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"No pude conectar con Qdrant: {exc}") from exc
    if response.status_code == 404:
        return False
    if not response.ok:
        raise HTTPException(status_code=502, detail=f"Qdrant respondió {response.status_code}: {response.text[:400]}")
    return True


def find_exact_article(article_number: str) -> list[dict]:
    """Find exact article headings in the small local RAG collection.

    Legal codes contain many repeated cross-references. For explicit article
    lookups, exact heading matches are more reliable than vector similarity.
    """
    heading = re.compile(
        rf"(?im)\bART[IÍ]CULO\s+{re.escape(article_number)}\s*(?:[º°o])?\s*[.\-—–:]"
    )
    matches = []
    records = {}
    offset = None
    while True:
        body = {"limit": 256, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        result = qdrant_request("POST", f"/collections/{COLLECTION}/points/scroll", json=body).get("result", {})
        points = result.get("points", [])
        for point in points:
            payload = point.get("payload") or {}
            text = payload.get("text", "")
            if not text:
                continue
            key = (payload.get("source"), payload.get("page"), payload.get("chunk_index"), text)
            records[key] = payload
        offset = result.get("next_page_offset")
        if not offset or not points:
            break

    # Rebuild the complete article from this document's ordered chunks. A long
    # article can continue in later chunks or pages without repeating its heading.
    article_heading = re.compile(r"(?im)\bART[IÍ]CULO\s+\d+\s*(?:[º°o])?\s*[.\-—–:]")
    ordered = sorted(
        records.values(),
        key=lambda item: (str(item.get("source", "")), int(item.get("page") or 0), int(item.get("chunk_index") or 0)),
    )
    for start_index, payload in enumerate(ordered):
        text = payload.get("text", "")
        found = heading.search(text)
        if not found:
            continue
        article_parts = [text[found.start():].strip()]
        for following in ordered[start_index + 1:]:
            if following.get("source") != payload.get("source"):
                break
            next_text = following.get("text", "").strip()
            next_heading = article_heading.search(next_text)
            if next_heading:
                article_parts.append(next_text[:next_heading.start()].strip())
                break
            article_parts.append(next_text)
            if sum(map(len, article_parts)) >= 16000:
                break
        full_text = "\n\n".join(part for part in article_parts if part).strip()
        key = (payload.get("source"), found.group(0).lower())
        if full_text and key not in {(m["payload"].get("source"), m["payload"].get("matched_heading")) for m in matches}:
            enriched = dict(payload)
            enriched["text"] = full_text
            enriched["matched_heading"] = found.group(0).lower()
            matches.append({"payload": enriched, "score": 1.0})
    return matches


def split_text(text: str) -> list[str]:
    text = re.sub(r"[\t\u00a0]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + (1 if text[boundary] == "\n" else 2 if text[boundary:boundary + 2] == ". " else 0)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


@app.get("/")
def home():
    return FileResponse(STATIC_FILE)


@app.get("/api/status")
def status():
    try:
        qdrant = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
        qdrant_ok = qdrant.ok or qdrant.status_code == 404
        count = qdrant.json().get("result", {}).get("points_count", 0) if qdrant.ok else 0
        ollama = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        ollama_ok = ollama.ok
        installed = [item.get("name") for item in ollama.json().get("models", [])] if ollama.ok else []
    except requests.RequestException:
        qdrant_ok = False
        ollama_ok = False
        count = 0
        installed = []
    return {
        "qdrant": qdrant_ok,
        "ollama": ollama_ok,
        "collection": COLLECTION,
        "points": count,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "models_installed": installed,
        "chat_provider": "n8n" if N8N_CHAT_WEBHOOK_URL else "direct",
        "ingestion_provider": "n8n" if N8N_INGEST_WEBHOOK_URL else "direct",
    }


@app.post("/api/documents")
async def ingest_document(file: UploadFile = File(...)):
    name = Path(file.filename or "document.pdf").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Por ahora la app recibe archivos PDF.")
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="El PDF supera el límite de 30 MB.")
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="No pude abrir el PDF. Comprobá que no esté dañado o protegido con contraseña.") from exc

    items = []
    for page_index, page in enumerate(document):
        page_text = page.get_text("text").strip()
        for chunk_index, chunk in enumerate(split_text(page_text)):
            items.append({"text": chunk, "page": page_index + 1, "chunk_index": chunk_index})
    if not items:
        raise HTTPException(status_code=422, detail="No encontré texto seleccionable en el PDF. Si es escaneado, hace falta OCR.")

    if N8N_INGEST_WEBHOOK_URL:
        try:
            response = requests.post(
                N8N_INGEST_WEBHOOK_URL,
                files={"data": (name, data, "application/pdf")},
                timeout=(10, 1800),
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"No pude enviar el PDF al workflow de n8n: {exc}") from exc
        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"El workflow de n8n respondió {response.status_code}: {response.text[:400]}",
            )
        return {
            "filename": name,
            "pages": len(document),
            "chunks": None,
            "indexed": True,
            "via_n8n": True,
        }

    point_ids = []
    try:
        for batch_start in range(0, len(items), 24):
            batch = items[batch_start:batch_start + 24]
            vectors = embed_texts([item["text"] for item in batch])
            ensure_collection(len(vectors[0]))
            points = []
            for item, vector in zip(batch, vectors):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "text": item["text"],
                        "source": name,
                        "page": item["page"],
                        "chunk_index": item["chunk_index"],
                        "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    },
                })
            qdrant_request("PUT", f"/collections/{COLLECTION}/points?wait=true", json={"points": points})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"No pude indexar el PDF: {exc}") from exc
    return {"filename": name, "pages": len(document), "chunks": len(items), "indexed": len(point_ids)}


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Escribí una pregunta.")
    if N8N_CHAT_WEBHOOK_URL:
        try:
            response = requests.post(
                N8N_CHAT_WEBHOOK_URL,
                json={
                    "action": "sendMessage",
                    "sessionId": request.session_id or uuid.uuid4().hex,
                    "chatInput": question,
                },
                timeout=(10, 900),
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"No pude conectar con el workflow de n8n: {exc}") from exc
        if not response.ok:
            raise HTTPException(
                status_code=502,
                detail=f"n8n respondió {response.status_code}: {response.text[:400]}",
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="n8n devolvió una respuesta que no pude interpretar.") from exc
        if isinstance(result, list):
            result = result[0] if result else {}
        if isinstance(result, dict) and isinstance(result.get("data"), dict):
            result = result["data"]
        answer = ""
        if isinstance(result, dict):
            answer = str(result.get("output") or result.get("text") or result.get("response") or "").strip()
        if not answer:
            raise HTTPException(status_code=502, detail="n8n no devolvió texto de respuesta.")
        return {"answer": answer, "sources": [], "session_id": request.session_id}
    if not collection_exists():
        return {"answer": "Todavía no hay documentos indexados. Subí un PDF para empezar.", "sources": []}
    article_match = re.search(r"(?i)\bart[ií]culo\s+(\d+)\b", question)
    hits = find_exact_article(article_match.group(1)) if article_match else []
    if not hits:
        vector = embed_texts([question])[0]
        result = qdrant_request(
            "POST",
            f"/collections/{COLLECTION}/points/query",
            json={"query": vector, "limit": 5, "with_payload": True},
        )
        hits = result.get("result", {}).get("points", [])
    if not hits:
        return {"answer": "Todavía no encuentro documentos cargados. Subí un PDF para empezar.", "sources": []}
    sources = []
    context_parts = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        text = payload.get("text", "").strip()
        if not text:
            continue
        source = {"filename": payload.get("source", "Documento"), "page": payload.get("page"), "score": round(hit.get("score", 0), 3)}
        sources.append(source)
        context_parts.append(f"[Fuente {rank}: {source['filename']}, página {source['page']}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)
    system = (
        "Sos el asistente del RAG-Lab. Respondé en español usando únicamente el CONTEXTO recuperado. "
        "Si el contexto no alcanza para responder, decilo claramente y no inventes datos. "
        "Cuando afirmes algo, mencioná la fuente y página indicadas entre corchetes. "
        "Si preguntan por el texto de un artículo, reproducí todo el texto disponible de ese artículo, sin resumirlo ni omitir su continuación. "
        "No agregues una nota diciendo que el texto está cortado salvo que el propio contexto termine claramente a mitad de una frase. "
        "Mantené la respuesta clara y directa."
    )
    response = ollama_request(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": f"{system}\n\nCONTEXTO:\n{context}"},
                {"role": "user", "content": question},
            ],
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096},
        },
    )
    answer = response.get("message", {}).get("content", "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="Ollama no devolvió una respuesta.")
    return {"answer": answer, "sources": sources}

