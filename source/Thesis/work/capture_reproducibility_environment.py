import hashlib
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from thesis_rag.canonical import canonical_json
from thesis_rag.config import load_config
from thesis_rag.detector import PromptGuardDetectorAdapter
from thesis_rag.embeddings import OllamaEmbeddingAdapter
from thesis_rag.generation import OllamaGeneratorAdapter

_ROOT = Path(__file__).resolve().parents[1]


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if (_ROOT / "artifacts/protocol-lock.json").exists():
        raise RuntimeError(
            "refusing to recapture environment after protocol lock creation"
        )
    config = load_config(_ROOT / "configs/main-draft.yaml")
    generation = config.generation
    generator_identity = OllamaGeneratorAdapter(
        model_id=generation.model_id,
        model_revision=generation.model_revision,
        expected_template_sha256=generation.chat_template_sha256,
        expected_quantization=generation.quantization,
        context_window=generation.context_window,
        max_new_tokens=generation.max_new_tokens,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        min_p=generation.min_p,
        seed=generation.seed,
        thinking=generation.thinking,
        endpoint=generation.endpoint,
        keep_alive=generation.keep_alive,
    ).verify_model()
    embedding = config.embedding
    embedding_adapter = OllamaEmbeddingAdapter(
        model_id=embedding.model_id,
        model_revision=embedding.model_revision,
        dimensions=embedding.dimensions,
        query_instruction=embedding.query_instruction,
        endpoint=embedding.endpoint,
        truncate=embedding.truncate,
        keep_alive=embedding.keep_alive,
    )
    embedding_adapter.verify_model()
    detector = config.detector
    detector_identity = PromptGuardDetectorAdapter(
        model_id=detector.model_id,
        model_revision=detector.model_revision,
        max_length=detector.max_length,
        overlap_tokens=detector.overlap_tokens,
        malicious_label=detector.malicious_label,
    ).verify_model()

    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    package_lock_path = (
        _ROOT / "artifacts/reproducibility/python-environment-lock.txt"
    )
    package_lock_contents = "\n".join(sorted(pip_freeze)) + "\n"
    _atomic_write(package_lock_path, package_lock_contents)

    runtime_path = (
        _ROOT / "artifacts/reproducibility/runtime-environment.json"
    )
    runtime = {
        "schema_version": "1.0.0",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "key_packages": {
            package: version(package)
            for package in (
                "numpy",
                "pydantic",
                "scipy",
                "statsmodels",
                "torch",
                "transformers",
            )
        },
        "python_environment_lock_sha256": _sha256_file(package_lock_path),
        "models": {
            "generator": generator_identity.model_dump(mode="json"),
            "embedding": {
                "model_id": embedding.model_id,
                "model_revision": embedding.model_revision,
                "dimensions": embedding.dimensions,
                "runtime_version": embedding_adapter.runtime_version(),
            },
            "detector": detector_identity.model_dump(mode="json"),
        },
        "test_data_scoring_performed": False,
    }
    _atomic_write(runtime_path, f"{canonical_json(runtime)}\n")
    print(
        canonical_json(
            {
                "status": "environment_captured_without_test_data_scoring",
                "runtime_path": runtime_path.relative_to(_ROOT).as_posix(),
                "runtime_sha256": _sha256_file(runtime_path),
                "package_lock_path": (
                    package_lock_path.relative_to(_ROOT).as_posix()
                ),
                "package_lock_sha256": _sha256_file(package_lock_path),
            }
        )
    )


if __name__ == "__main__":
    main()
