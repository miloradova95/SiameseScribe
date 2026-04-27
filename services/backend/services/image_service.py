import random as _random
import logging
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import httpx
from fastapi import UploadFile
from sqlmodel import Session, select
from services.backend.sqlDB.images import Image
from services.backend.sqlDB.patches import Patch
from services.ML.app.chroma_client import get_chroma_client, get_or_create_collection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "data" / "dataset" / "preprocessed" / "TEMP"
ML_API_URL = os.getenv("ML_API_URL", "http://127.0.0.1:8001").rstrip("/")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "patches_v1")
logger = logging.getLogger(__name__)


class UploadPipelineError(RuntimeError):
    def __init__(self, step: str, message: str):
        super().__init__(f"{step}: {message}")
        self.step = step
        self.message = message


def get_all(session: Session) -> list[Image]:
    return session.exec(select(Image)).all()


def get_by_id(session: Session, image_id: int) -> Image | None:
    return session.get(Image, image_id)


def get_id_by_filename(session: Session, file_name: str) -> int | None:
    result = session.exec(select(Image).where(Image.fileName == file_name)).first()
    return result.id if result else None


def get_random(session: Session) -> Image | None:
    all_ids = session.exec(select(Image.id)).all()
    if not all_ids:
        return None
    return session.get(Image, _random.choice(all_ids))


def create(session: Session, file_name: str, file_path: str, group: str | None = None) -> Image:
    image = Image(fileName=file_name, filePath=file_path, group=group)
    session.add(image)
    session.commit()
    session.refresh(image)
    return image


def save_upload(session: Session, file: UploadFile, group: str | None = None) -> Image:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    extension = os.path.splitext(file.filename or "")[1]
    unique_name = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    logger.info(
        "Saving uploaded image",
        extra={
            "original_filename": file.filename,
            "group": group,
            "target_path": file_path,
        },
    )

    with open(file_path, "wb") as destination:
        shutil.copyfileobj(file.file, destination)

    image = create(session, file_name=file.filename, file_path=file_path, group=group)
    logger.info(
        "Upload saved",
        extra={
            "image_id": image.id,
            "original_filename": file.filename,
            "stored_path": file_path,
        },
    )
    return image


def _relative_to_workspace(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.parent)).replace("\\", "/")


def _iter_ml_api_urls() -> list[str]:
    urls = [ML_API_URL]
    parts = urlsplit(ML_API_URL)

    # Uvicorn commonly binds to IPv4 loopback only during local development.
    # On Windows, "localhost" can resolve to IPv6 first, so keep a 127.0.0.1
    # fallback for the same port when the env var uses localhost.
    if parts.hostname == "localhost":
        netloc = "127.0.0.1"
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        urls.append(urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/"))

    seen: set[str] = set()
    ordered_urls: list[str] = []
    for url in urls:
        if url not in seen:
            ordered_urls.append(url)
            seen.add(url)
    return ordered_urls


def _post_to_ml_api(endpoint: str, *, json: dict, timeout: float) -> httpx.Response:
    last_error: httpx.HTTPError | None = None

    for base_url in _iter_ml_api_urls():
        try:
            response = httpx.post(f"{base_url}{endpoint}", json=json, timeout=timeout)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning(
                "ML backend request failed; trying next candidate if available",
                extra={"ml_api_url": base_url, "endpoint": endpoint},
            )

    if last_error is None:
        raise RuntimeError("No ML API URL candidates were configured")
    raise last_error


def segment_upload_image(session: Session, image: Image) -> list[Patch]:
    # Step 1: resolve the uploaded image path and log which file is being sent
    # to the ML segmentation service.
    absolute_image_path = resolve_file_path(image).resolve()
    logger.info(
        "Starting segmentation for uploaded image",
        extra={
            "image_id": image.id,
            "image_path": str(absolute_image_path),
            "ml_api_url": ML_API_URL,
        },
    )

    # Step 2: call the ML backend and convert network / HTTP failures into a
    # pipeline-specific error that keeps the failing step name.
    try:
        response = _post_to_ml_api(
            "/segment",
            json={"image_path": str(absolute_image_path)},
            timeout=120.0,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or str(exc)
        logger.exception(
            "Segmentation request failed",
            extra={"image_id": image.id, "status_code": exc.response.status_code},
        )
        raise UploadPipelineError("segment_upload_image", detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("Segmentation request could not reach ML backend", extra={"image_id": image.id})
        raise UploadPipelineError("segment_upload_image", str(exc)) from exc

    # Step 3: unpack the returned patch metadata and map it into backend Patch
    # rows that are ready to be persisted.
    patches_payload = response.json().get("patches", [])
    logger.info(
        "Segmentation finished",
        extra={"image_id": image.id, "patch_count": len(patches_payload)},
    )
    patches: list[Patch] = []

    for patch_data in patches_payload:
        patch_relative_path = patch_data["patch_path"]
        patch_absolute_path = (PROJECT_ROOT / patch_relative_path).resolve()
        bbox = patch_data["bbox"]

        patches.append(Patch(
            source_image_id=image.id,
            file_path=_relative_to_workspace(patch_absolute_path),
            bbox={
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"],
            },
            group=image.group,
            codex=None,
            pen_flourishing_percent=None,
        ))

    # Step 4: persist the empty result explicitly when the ML service found no
    # valid patches so the image still has a known processed state.
    if not patches:
        image.patches = []
        session.add(image)
        session.commit()
        session.refresh(image)
        logger.warning("Segmentation returned no patches", extra={"image_id": image.id})
        return []

    # Step 5: store the new patches, refresh them to obtain database IDs, and
    # link those IDs back onto the source image record.
    session.add_all(patches)
    session.commit()

    for patch in patches:
        session.refresh(patch)

    image.patches = [patch.id for patch in patches]
    session.add(image)
    session.commit()
    session.refresh(image)
    logger.info(
        "Stored segmented patches",
        extra={"image_id": image.id, "patch_ids": image.patches, "patch_count": len(patches)},
    )

    return patches


def embed_new_patches(patches: list[Patch]) -> list[dict]:
    # Step 1: short-circuit early when segmentation produced nothing to embed.
    if not patches:
        logger.warning("Skipping embedding because no patches were provided")
        return []

    # Step 2: collect stored patch paths and send them to the ML embedding
    # endpoint so the backend does not need to load model weights itself.
    patch_paths = [patch.file_path for patch in patches]
    logger.info(
        "Starting embedding for uploaded patches",
        extra={"patch_count": len(patch_paths), "first_patch_path": patch_paths[0]},
    )

    try:
        response = _post_to_ml_api(
            "/embed_patches",
            json={"patch_paths": patch_paths},
            timeout=120.0,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text or str(exc)
        logger.exception(
            "Embedding request failed",
            extra={"patch_count": len(patch_paths), "status_code": exc.response.status_code},
        )
        raise UploadPipelineError("embed_new_patches", detail) from exc
    except httpx.HTTPError as exc:
        logger.exception("Embedding request could not reach ML backend", extra={"patch_count": len(patch_paths)})
        raise UploadPipelineError("embed_new_patches", str(exc)) from exc

    # Step 3: return the ML response payload as-is so the next stage can match
    # vectors back to persisted patches by path.
    embeddings = response.json().get("embeddings", [])
    logger.info(
        "Embedding finished",
        extra={"patch_count": len(patch_paths), "embedding_count": len(embeddings)},
    )
    return embeddings


def upload_new_embeddings(patches: list[Patch], embeddings: list[dict]) -> None:
    # Step 1: stop early if one of the previous pipeline stages produced no data.
    if not patches or not embeddings:
        logger.warning(
            "Skipping Chroma upload because patches or embeddings are missing",
            extra={"patch_count": len(patches), "embedding_count": len(embeddings)},
        )
        return

    patch_by_path = {patch.file_path: patch for patch in patches}
    ordered_embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # Step 2: align returned embeddings with persisted patches and build the
    # Chroma payload (IDs, vectors, metadata) in a deterministic order.
    for item in embeddings:
        patch = patch_by_path.get(item["patch_path"])
        if patch is None:
            logger.warning(
                "Embedding returned for unknown patch path",
                extra={"patch_path": item.get("patch_path")},
            )
            continue

        patch_path = Path(patch.file_path)
        ordered_embeddings.append(item["vector"])
        ids.append(patch_path.name)
        metadatas.append({
            "source_image_id": patch.source_image_id,
            "group": patch.group or "",
            "codex": patch.codex or "",
            "x": patch.bbox["x"],
            "y": patch.bbox["y"],
            "width": patch.bbox["width"],
            "height": patch.bbox["height"],
            "pen_flourishing_percent": patch.pen_flourishing_percent if patch.pen_flourishing_percent is not None else -1.0,
            "file_path": patch.file_path,
        })

    if not ids:
        logger.warning("No embeddings matched persisted patches; skipping Chroma upsert")
        return

    # Step 3: upsert the matched embeddings into Chroma so reprocessing the same
    # patch filenames updates existing vectors instead of duplicating them.
    chroma_path = str(PROJECT_ROOT / "data" / "chromaDB" / "data" / "chroma_store")
    logger.info(
        "Uploading embeddings to Chroma",
        extra={
            "collection": CHROMA_COLLECTION,
            "chroma_path": chroma_path,
            "embedding_count": len(ids),
        },
    )

    try:
        client = get_chroma_client(chroma_path)
        collection = get_or_create_collection(client, CHROMA_COLLECTION)
        collection.upsert(ids=ids, embeddings=ordered_embeddings, metadatas=metadatas)
    except Exception as exc:
        logger.exception(
            "Failed to upload embeddings to Chroma",
            extra={"collection": CHROMA_COLLECTION, "embedding_count": len(ids)},
        )
        raise UploadPipelineError("upload_new_embeddings", str(exc)) from exc

    logger.info(
        "Uploaded embeddings to Chroma",
        extra={"collection": CHROMA_COLLECTION, "embedding_count": len(ids)},
    )


def save_and_process_upload(session: Session, file: UploadFile, group: str | None = None) -> Image:
    # Step 1: record the start of the full upload pipeline for this file.
    logger.info(
        "Starting upload pipeline",
        extra={"original_filename": file.filename, "group": group},
    )

    try:
        # Step 2: store the uploaded image file and create its database row.
        image = save_upload(session, file=file, group=group)
        logger.info("Upload pipeline saved image", extra={"image_id": image.id})

        # Step 3: segment the image and persist the generated patch rows.
        patches = segment_upload_image(session, image)
        logger.info("Upload pipeline segmented image", extra={"image_id": image.id, "patch_count": len(patches)})

        # Step 4: request embeddings for the new patches from the ML backend.
        embeddings = embed_new_patches(patches)
        logger.info(
            "Upload pipeline embedded patches",
            extra={"image_id": image.id, "embedding_count": len(embeddings)},
        )

        # Step 5: store the embeddings in Chroma and finish with the original
        # image row as the API response payload.
        upload_new_embeddings(patches, embeddings)
        logger.info(
            "Upload pipeline completed successfully",
            extra={
                "image_id": image.id,
                "patch_count": len(patches),
                "embedding_count": len(embeddings),
            },
        )
        return image
    # Step 6: preserve explicit pipeline errors and wrap only truly unexpected
    # failures so the caller still gets the step name when possible.
    except UploadPipelineError:
        logger.exception("Upload pipeline failed")
        raise
    except Exception as exc:
        logger.exception("Upload pipeline failed unexpectedly", extra={"original_filename": file.filename})
        raise UploadPipelineError("save_and_process_upload", str(exc)) from exc


def resolve_file_path(image: Image) -> Path:
    return PROJECT_ROOT.parent / image.filePath.replace("\\", "/")
