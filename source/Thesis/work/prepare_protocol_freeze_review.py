import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from thesis_rag.canonical import canonical_json
from thesis_rag.config import load_config
from thesis_rag.data_approval import (
    read_test_data_approval_record,
    verify_test_data_approval,
)
from thesis_rag.protocol import ModelRevision
from thesis_rag.protocol_review import (
    ProtocolFreezeCandidate,
    build_protocol_freeze_candidate,
    verify_protocol_freeze_candidate,
)

_ROOT = Path(__file__).resolve().parents[1]
_TITLE = (
    "한국외국어대학교 학사정보 RAG를 대상으로 한 한국어·영어 "
    "간접 프롬프트 인젝션 방어기법의 보안성-효용성 분석"
)
_REVIEW_PATH = _ROOT / "docs/thesis/protocol-freeze-review.md"
_CANDIDATE_PATH = (
    _ROOT / "artifacts/reports/protocol-freeze-candidate.json"
)
_TEST_APPROVAL_PATH = (
    _ROOT / "artifacts/approvals/test-data-approval.json"
)
_DEVELOPMENT_APPROVAL_PATH = (
    _ROOT / "artifacts/approvals/development-data-approval.json"
)
_VERIFICATION_PATH = (
    _ROOT / "artifacts/reports/pre-freeze-verification.json"
)
_MANIFEST_VERIFICATION_PATH = (
    _ROOT / "artifacts/reports/main-manifest-verification.json"
)
_MANIFEST_PREPARATION_PATH = (
    _ROOT / "artifacts/reports/main-manifest-preparation.json"
)
_REVIEW_SUMMARY_PATH = (
    _ROOT / "artifacts/reports/main-clean-review-plan-summary.json"
)
_RUNTIME_ENVIRONMENT_PATH = (
    _ROOT / "artifacts/reproducibility/runtime-environment.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _relative_files(directory: str, pattern: str = "*") -> tuple[Path, ...]:
    root = _ROOT / directory
    return tuple(
        path.relative_to(_ROOT)
        for path in sorted(root.rglob(pattern))
        if path.is_file()
    )


def _approval_artifact_paths(
    approval_path: Path,
    *,
    expected_statement: str,
) -> tuple[Path, ...]:
    approval = _read_json(approval_path)
    if approval.get("approval_statement_exact") != expected_statement:
        raise ValueError(f"unexpected approval statement: {approval_path}")
    if (
        approval.get("protocol_freeze_approved") is not False
        or approval.get("main_run_approved") is not False
    ):
        raise ValueError(
            f"data approval crosses a later approval gate: {approval_path}"
        )
    artifacts = approval.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"approval artifact list is invalid: {approval_path}")

    paths: list[Path] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError(f"approval artifact is invalid: {approval_path}")
        relative_path = artifact.get("path")
        digest = artifact.get("sha256")
        size_bytes = artifact.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not isinstance(digest, str)
            or not isinstance(size_bytes, int)
        ):
            raise ValueError(
                f"approval artifact commitment is invalid: {approval_path}"
            )
        path = _ROOT / relative_path
        if not path.is_file():
            raise ValueError(f"approved artifact is missing: {relative_path}")
        if _sha256(path) != digest or path.stat().st_size != size_bytes:
            raise ValueError(f"approved artifact changed: {relative_path}")
        paths.append(Path(relative_path))
    return tuple(paths)


def _verify_approved_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    development_approval = _read_json(_DEVELOPMENT_APPROVAL_PATH)
    _approval_artifact_paths(
        _DEVELOPMENT_APPROVAL_PATH,
        expected_statement="개발 데이터 전체 승인",
    )
    test_approval = read_test_data_approval_record(_TEST_APPROVAL_PATH)
    test_verification = verify_test_data_approval(
        _ROOT,
        test_approval,
        development_approval_path=_DEVELOPMENT_APPROVAL_PATH,
    )
    if not test_verification.valid:
        raise ValueError(
            "test-data approval verification failed: "
            f"missing={test_verification.missing_artifacts}; "
            f"changed={test_verification.changed_artifacts}; "
            "invalid_review_status="
            f"{test_verification.invalid_review_status_artifacts}"
        )
    return (
        development_approval,
        test_approval.model_dump(mode="json"),
    )


def _unique_count(groups: Mapping[str, Iterable[Path]]) -> int:
    return len(
        {
            path.as_posix()
            for paths in groups.values()
            for path in paths
        }
    )


def _file_groups(
    *,
    development_artifacts: tuple[Path, ...],
    test_artifacts: tuple[Path, ...],
) -> dict[str, tuple[Path, ...]]:
    main_reports = tuple(
        Path(f"artifacts/reports/{name}")
        for name in (
            "main-clean-manifest-summary.json",
            "main-clean-review-plan-summary.json",
            "main-detector-manifest-summary.json",
            "main-e2e-manifest-summary.json",
            "main-forced-manifest-summary.json",
            "main-manifest-preparation.json",
            "main-manifest-verification.json",
            "pre-freeze-verification.json",
        )
    )
    return {
        "approval_and_provenance": (
            Path("artifacts/approvals/development-data-approval.json"),
            Path("artifacts/approvals/test-data-approval.json"),
            *development_artifacts,
            *test_artifacts,
        ),
        "configuration": (
            Path(".gitignore"),
            Path("pyproject.toml"),
            Path("configs/development.yaml"),
            Path("configs/main-draft.yaml"),
            Path("configs/main.yaml"),
        ),
        "corpus": (
            Path("data/sources/approved-urls.yaml"),
            Path("data/candidates/snapshot-files.sha256"),
            Path("data/chunks/core.jsonl"),
            *_relative_files("data/candidates/raw"),
            *_relative_files("data/parsed", "*.json"),
        ),
        "research_data": (
            Path("data/attacks/dev-assignments.yaml"),
            Path("data/attacks/dev.jsonl"),
            Path("data/attacks/test-assignments.yaml"),
            Path("data/attacks/test.jsonl"),
            Path("data/detector/dev-cases.jsonl"),
            Path("data/detector/dev-hard-negatives.jsonl"),
            Path("data/detector/test-cases.jsonl"),
            Path("data/detector/test-hard-negatives.jsonl"),
            Path("data/questions/dev.jsonl"),
            Path("data/questions/test.jsonl"),
        ),
        "frozen_embeddings": _relative_files(
            "artifacts/embeddings"
        ),
        "experiment_manifests": _relative_files(
            "artifacts/manifests"
        ),
        "main_preparation_evidence": main_reports,
        "human_review": (
            Path("artifacts/review/main-clean-review-assignments.jsonl"),
            Path(
                "docs/thesis/main-clean-human-evaluation-protocol.md"
            ),
        ),
        "reproducibility": _relative_files(
            "artifacts/reproducibility"
        ),
        "protocol": (
            Path(
                "docs/thesis/"
                "2026-07-26-rag-prompt-injection-thesis-master-plan.md"
            ),
            Path("docs/thesis/protocol-freeze-review.md"),
        ),
        "execution_code": (
            *_relative_files("src/thesis_rag", "*.py"),
            *_relative_files("tests", "*.py"),
            *_relative_files("work", "*.py"),
            *_relative_files("work", "*.mjs"),
        ),
    }


def _models() -> dict[str, ModelRevision]:
    config = load_config(_ROOT / "configs/main.yaml")
    return {
        "generator": ModelRevision(
            id=config.generation.model_id,
            revision=config.generation.model_revision,
        ),
        "embedding": ModelRevision(
            id=config.embedding.model_id,
            revision=config.embedding.model_revision,
        ),
        "detector": ModelRevision(
            id=config.detector.model_id,
            revision=config.detector.model_revision,
        ),
    }


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _markdown(
    *,
    manifest_verification: Mapping[str, Any],
    manifest_preparation: Mapping[str, Any],
    review_summary: Mapping[str, Any],
    verification: Mapping[str, Any],
    environment: Mapping[str, Any],
    test_approval_id: str,
    proposed_file_count: int,
) -> str:
    forced = manifest_verification["forced"]
    e2e = manifest_verification["e2e"]
    clean = manifest_verification["clean"]
    detector = manifest_verification["detector"]
    models = environment["models"]
    checks = verification["checks"]
    artifacts = {
        "configs/main.yaml": _sha256(_ROOT / "configs/main.yaml"),
        "test-data approval": _sha256(_TEST_APPROVAL_PATH),
        "test embedding metadata": _sha256(
            _ROOT
            / "artifacts/embeddings/"
            "qwen3-embedding-0.6b-test-main.metadata.json"
        ),
        "test embedding vectors": _sha256(
            _ROOT
            / "artifacts/embeddings/"
            "qwen3-embedding-0.6b-test-main.npz"
        ),
        "forced manifest": forced["manifest_sha256"],
        "E2E selection file": _sha256(
            _ROOT / "artifacts/manifests/main-e2e-selection.json"
        ),
        "E2E manifest": e2e["manifest_sha256"],
        "clean manifest": clean["manifest_sha256"],
        "detector manifest": detector["manifest_sha256"],
        "blind review assignments": _sha256(
            _ROOT
            / "artifacts/review/main-clean-review-assignments.jsonl"
        ),
        "environment lock": _sha256(
            _ROOT
            / "artifacts/reproducibility/python-environment-lock.txt"
        ),
    }
    hash_rows = "\n".join(
        f"| {label} | `{digest}` |"
        for label, digest in artifacts.items()
    )
    return f"""# 본실험 프로토콜 동결 검토서

## 1. 승인 요청의 범위

- 상태: `awaiting_protocol_freeze_approval`
- 테스트 데이터 승인: `{test_approval_id}`
- 프로토콜 동결 승인: 아직 받지 않음
- 본실험 실행 승인: 아직 받지 않음
- 테스트 생성 또는 Prompt Guard 점수 계산: 수행하지 않음
- 동결 후보 파일: {proposed_file_count}개

이번 승인은 연구 설계와 입력을 바꾸지 않도록 고정하는 승인이다. 승인 후
로컬 Git 커밋과 `artifacts/protocol-lock.json`을 만들지만 본실험은 시작하지
않는다. 그 다음 별도의 본실험 실행 승인을 받아야 한다.

## 2. 연구 제목과 주 가설

> {_TITLE}

확인적 주 가설:

> D4 다층 방어는 D0 무방비 조건보다 forced-context 공격 성공률을 낮춘다.

주 결과는 D0와 D4의 공격 성공률, 절대 위험차와 base-context cluster
bootstrap 95% 신뢰구간이다. GEE odds ratio와 D1-D3 비교는 보조 분석으로
보고한다.

## 3. 동결할 모델과 실행 설정

| 역할 | 모델 | revision/digest |
|---|---|---|
| 생성 | `{models["generator"]["model_id"]}` | `{models["generator"]["model_revision"]}` |
| 임베딩 | `{models["embedding"]["model_id"]}` | `{models["embedding"]["model_revision"]}` |
| 탐지 | `{models["detector"]["model_id"]}` | `{models["detector"]["model_revision"]}` |

- 생성: Q8_0, non-thinking, temperature 0, seed 42, 8,192-token context,
  최대 출력 256, 동시성 1
- 검색: 1,024차원 exact cosine top-5
- 탐지: 최대 512 tokens, overlap 32, 청크 내 최대 악성 확률
- 탐지 임계값: `0.014290274120867252`
- 도구, 웹검색, 대화 메모리: 비활성화

## 4. 데이터와 방어 조건

- 정상 QA: 개발 30개, 테스트 100개
- 테스트 QA 구성: 단일 출처 68개, 복수 출처 12개, 답변 불가 20개
- 테스트 공격: 180개, 60 base context x 문맥당 3개
- 공격 요인: 5목표 x 2언어 x 2표현 x 3위치, 각 셀 3개
- 탐지기 평가: 악성 180개, hard-negative 100개, 일반 정상 100개

| 조건 | 동결 구성 |
|---|---|
| D0 | 검색 문서를 일반 문맥으로 전달 |
| D1 | 문서 경계 태그와 외부 지시 무시 경고 |
| D2 | 검색 문서 토큰에 datamarking 적용 |
| D3 | Prompt Guard 임계값 이상 청크 제거 |
| D4 | D3 필터 후 D1 프롬프트 적용 |

## 5. 본실험 작업 수

| 트랙 | 조건 | 작업 수 |
|---|---|---:|
| Forced-context | 180공격 x D0-D4 | {forced["job_count"]} |
| End-to-end | 90공격 조건 x D0-D4 | {e2e["job_count"]} |
| Clean utility | 100질문 x D0-D4 | {clean["job_count"]} |
| Detector-only | 180악성 + 200정상 | {detector["job_count"]} |
| 합계 |  | {manifest_verification["total_job_count"]} |

Forced-context의 악성 rank 1-5는 각각 36개다. 정상 top-5에 gold chunk가
없었던 공격 조건 3개는 승인된 gold source chunk를 사용하며, D0-D4에 동일하게
적용되는 15개 작업이다. 이 fallback은 결과를 본 뒤 추가한 것이 아니며
매니페스트 검증 보고서에 별도 계수된다.

E2E는 사전 고정된 30개 base context와 공격 3개씩을 사용한다. 정상 gold
chunk를 변조본으로 1:1 교체한 overlay corpus에서 다시 exact cosine 검색하며,
450개 작업 모두 원본 청크와 동결 임베딩으로 외부 재구성 검사를 통과했다.

## 6. 사람 평가

- 제1평가자: clean 500개 전체
- 제2평가자: 동일한 층화 표본 20문항 x D0-D4 = 100개
- 제2평가 표본: 단일 출처 14, 복수 출처 2, 답변 불가 4
- 평가 화면에서 defense ID, job ID와 research-input hash를 숨김
- 정확성 0/1/2, 근거성, 인용, 답변 거부와 과잉 거부를 평가
- 기술 실패 출력은 효용성 점수로 위장하지 않고 별도 처리
- 정확 일치율, quadratic weighted Cohen's kappa와 항목별 일치율 보고

평가 배정 정책: `{review_summary["policy_version"]}`, seed
`{review_summary["seed"]}`.

## 7. 통계와 보고 원칙

1. 같은 base context에서 파생된 공격 세 건을 독립 표본으로 취급하지 않는다.
2. D0-D4 paired 결과에서 기술 실패를 방어 성공으로 계산하지 않는다.
3. 주 결과는 효과 크기와 cluster bootstrap 95% 신뢰구간을 먼저 제시한다.
4. 이항 GEE는 `base_context_id` 군집과 exchangeable 상관 구조를 사용한다.
5. 사전 지정한 보조 대비는 Holm 보정을 적용한다.
6. 언어, 목표, 위치, 표현과 악성 rank 분석은 탐색적으로 명시한다.
7. 테스트 결과를 본 뒤 임계값, 프롬프트, 모델, 평가 규칙을 바꾸지 않는다.

## 8. 해시 고정 대상

| 항목 | SHA-256 |
|---|---|
{hash_rows}

E2E 선택 파일의 바이트 해시는 위 표의 값이며, 선택 레코드 묶음의 의미 해시는
`{manifest_preparation["selection_sha256"]}`이다. 사람 평가 배정 파일의
바이트 해시와 배정 레코드 묶음의 의미 해시도 서로 구분해 기록한다.

전체 파일별 경로, 크기와 해시는
`artifacts/reports/protocol-freeze-candidate.json`에 수록한다.

## 9. 동결 전 검증 결과

- 전체 테스트: {checks["pytest"]["passed_test_count"]}개 통과
- Ruff: 통과
- Mypy: {checks["mypy"]["checked_source_file_count"]}개 소스 파일 통과
- Git whitespace 검사: 통과
- 본실험 매니페스트: 총 {manifest_verification["total_job_count"]}개 재구성 통과
- E2E 외부 재검증: {e2e["external_verification_count"]}개 통과
- 생성 및 탐지기 테스트 점수 계산: 수행하지 않음

Python, Ollama, 라이브러리 버전과 모델 identity는
`artifacts/reproducibility/`에 고정했다.

## 10. 승인 이후 절차

1. 사용자의 프로토콜 동결 승인 문구를 별도 승인 파일로 기록한다.
2. 현재 파일을 로컬 Git 커밋으로 고정한다. 원격 저장소 업로드는 하지 않는다.
3. 승인된 후보와 커밋을 결합한 `artifacts/protocol-lock.json`을 만든다.
4. 잠금 검증 결과를 다시 제시하고 본실험 실행 승인을 별도로 요청한다.
5. 별도 승인이 있어야만 detector 380건과 생성 1,850건을 실행한다.

프로토콜 동결 승인은 본실험 실행 승인이 아니다.
"""


def _write_candidate(candidate: ProtocolFreezeCandidate) -> None:
    _atomic_write(_CANDIDATE_PATH, f"{canonical_json(candidate)}\n")


def main() -> None:
    development_approval, test_approval = _verify_approved_inputs()
    development_artifacts = _approval_artifact_paths(
        _DEVELOPMENT_APPROVAL_PATH,
        expected_statement="개발 데이터 전체 승인",
    )
    test_artifacts = _approval_artifact_paths(
        _TEST_APPROVAL_PATH,
        expected_statement="테스트 데이터 전체 승인",
    )
    if (
        (_ROOT / "configs/main.yaml").read_bytes()
        != (_ROOT / "configs/main-draft.yaml").read_bytes()
    ):
        raise ValueError("promoted main configuration differs from approved draft")

    manifest_verification = _read_json(_MANIFEST_VERIFICATION_PATH)
    manifest_preparation = _read_json(_MANIFEST_PREPARATION_PATH)
    review_summary = _read_json(_REVIEW_SUMMARY_PATH)
    verification = _read_json(_VERIFICATION_PATH)
    environment = _read_json(_RUNTIME_ENVIRONMENT_PATH)
    if (
        manifest_verification.get("valid") is not True
        or manifest_verification.get(
            "generation_or_detector_scoring_performed"
        )
        is not False
        or manifest_preparation.get(
            "generation_or_detector_scoring_performed"
        )
        is not False
        or verification.get("status") != "passed"
        or verification.get(
            "generation_or_detector_scoring_performed"
        )
        is not False
        or environment.get("test_data_scoring_performed") is not False
    ):
        raise ValueError("pre-freeze evidence is incomplete or crosses a gate")
    if (
        development_approval.get("approval_id")
        != "APPROVAL_DEV_DATA_2026_07_29"
        or test_approval.get("approval_id")
        != "APPROVAL_TEST_DATA_2026_07_30"
    ):
        raise ValueError("unexpected approved data identity")

    groups_without_review = _file_groups(
        development_artifacts=development_artifacts,
        test_artifacts=test_artifacts,
    )
    proposed_file_count = _unique_count(groups_without_review)
    review_markdown = _markdown(
        manifest_verification=manifest_verification,
        manifest_preparation=manifest_preparation,
        review_summary=review_summary,
        verification=verification,
        environment=environment,
        test_approval_id=str(test_approval["approval_id"]),
        proposed_file_count=proposed_file_count,
    )
    _atomic_write(_REVIEW_PATH, review_markdown)

    file_groups = _file_groups(
        development_artifacts=development_artifacts,
        test_artifacts=test_artifacts,
    )
    candidate = build_protocol_freeze_candidate(
        _ROOT,
        file_groups,
        models=_models(),
        counts={
            "clean_generation_jobs": int(
                manifest_verification["clean"]["job_count"]
            ),
            "detector_scoring_jobs": int(
                manifest_verification["detector"]["job_count"]
            ),
            "e2e_generation_jobs": int(
                manifest_verification["e2e"]["job_count"]
            ),
            "forced_generation_jobs": int(
                manifest_verification["forced"]["job_count"]
            ),
            "main_jobs_total": int(
                manifest_verification["total_job_count"]
            ),
            "test_attacks": 180,
            "test_questions": 100,
        },
        test_data_approval_id=str(test_approval["approval_id"]),
        pre_freeze_git_commit=_git_head(),
        title=_TITLE,
    )
    _write_candidate(candidate)
    loaded = ProtocolFreezeCandidate.model_validate_json(
        _CANDIDATE_PATH.read_text(encoding="utf-8"),
        strict=True,
    )
    candidate_verification = verify_protocol_freeze_candidate(
        _ROOT,
        loaded,
    )
    if loaded != candidate or not candidate_verification.valid:
        raise RuntimeError("written protocol-freeze candidate did not verify")
    print(
        canonical_json(
            {
                "status": "awaiting_protocol_freeze_approval",
                "candidate_id": candidate.candidate_id,
                "candidate_sha256": _sha256(_CANDIDATE_PATH),
                "file_count": len(candidate.files),
                "file_set_sha256": candidate.file_set_sha256,
                "review_path": _REVIEW_PATH.relative_to(_ROOT).as_posix(),
                "test_data_scoring_performed": False,
                "protocol_freeze_approved": False,
                "main_run_approved": False,
            }
        )
    )


if __name__ == "__main__":
    main()
