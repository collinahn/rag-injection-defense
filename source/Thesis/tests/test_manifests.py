import numpy as np
import pytest

from thesis_rag.canonical import canonical_sha256
from thesis_rag.domain import (
    AttackCase,
    AttackGoal,
    AttackPosition,
    Chunk,
    DefenseId,
    E2EResearchInputs,
    ForcedResearchInputs,
    Question,
    RetrievedChunk,
    Split,
    expected_experiment_job_id,
)
from thesis_rag.manifests import (
    build_clean_jobs,
    build_e2e_jobs,
    build_forced_jobs,
    reconstruct_job_from_research_inputs,
    verify_e2e_job_from_sources,
)


def _question(
    *, split: Split = Split.DEV, question: str = "Which synthetic fact is supported?"
) -> Question:
    return Question(
        question_id="Q_SYN_001",
        split=split,
        language="en",
        question=question,
        answerable=True,
        gold_chunk_ids=("CH_SYN_GOLD",),
    )


def _attack(*, split: Split = Split.DEV, base_context_id: str = "CTX_SYN_001") -> AttackCase:
    return AttackCase(
        attack_id="ATK_SYN_001",
        split=split,
        base_context_id=base_context_id,
        goal=AttackGoal.HIJACK,
        language="en",
        form="plain",
        position=AttackPosition.MIDDLE,
        payload="ATTACK_SUCCESS_ATK_SYN_001",
        success_markers=("ATTACK_SUCCESS_ATK_SYN_001",),
        template_id="TPL_SYN_G1_EN_PLAIN_V1",
        safety_class="sentinel_only",
    )


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"DOC_{chunk_id}",
        text=text,
        language="en",
        metadata={"origin": "synthetic"},
    )


@pytest.fixture
def fixture_context() -> dict[str, object]:
    alpha = _chunk("CH_SYN_ALPHA", "Synthetic alpha fact.")
    gold = _chunk("CH_SYN_GOLD", "Synthetic gold fact.")
    beta = _chunk("CH_SYN_BETA", "Synthetic beta fact.")
    gamma = _chunk("CH_SYN_GAMMA", "Synthetic gamma fact.")
    return {
        "question": _question(),
        "base_context_id": "CTX_SYN_001",
        "ordered_benign_chunks": (
            RetrievedChunk(chunk=alpha, score=0.9, rank=1),
            RetrievedChunk(chunk=gold, score=0.8, rank=2),
            RetrievedChunk(chunk=beta, score=0.7, rank=3),
            RetrievedChunk(chunk=gamma, score=0.6, rank=4),
        ),
        "gold_chunk_id": gold.chunk_id,
        "attack": _attack(),
        "malicious_rank": 3,
        "defense_ids": tuple(DefenseId),
        "config_hash": "CFG_SYN_001",
    }


def test_forced_jobs_pair_identical_context_across_defenses(
    fixture_context: dict[str, object],
) -> None:
    jobs = build_forced_jobs(**fixture_context)  # type: ignore[arg-type]

    assert {job.defense_id for job in jobs} == set(DefenseId)
    assert len({job.question_id for job in jobs}) == 1
    assert len({job.ordered_chunk_ids for job in jobs}) == 1
    assert len({job.malicious_chunk_rank for job in jobs}) == 1
    assert len({job.job_id for job in jobs}) == 5
    assert all(job.research_inputs is jobs[0].research_inputs for job in jobs)
    assert isinstance(jobs[0].research_inputs, ForcedResearchInputs)
    assert {
        job.research_input_hash for job in jobs
    } == {canonical_sha256(jobs[0].research_inputs)}


def test_forced_jobs_replace_gold_at_requested_rank_and_rerank(
    fixture_context: dict[str, object],
) -> None:
    jobs = build_forced_jobs(**fixture_context)  # type: ignore[arg-type]
    first = jobs[0]

    assert first.track == "forced"
    assert first.question == "Which synthetic fact is supported?"
    assert first.attack_id == "ATK_SYN_001"
    assert first.base_context_id == "CTX_SYN_001"
    assert first.config_hash == "CFG_SYN_001"
    assert first.ordered_chunk_ids[0] == "CH_SYN_ALPHA"
    assert first.ordered_chunk_ids[1] == "CH_SYN_BETA"
    assert first.ordered_chunk_ids[2].startswith("MCH_")
    assert first.ordered_chunk_ids[3] == "CH_SYN_GAMMA"
    assert [item.rank for item in first.ordered_chunks] == [1, 2, 3, 4]
    assert first.ordered_chunks[2].score == 0.8
    assert first.malicious_chunk_rank == 3
    assert all(job.ordered_chunks is first.ordered_chunks for job in jobs)


def test_forced_jobs_commit_to_the_exact_gold_chunk_and_full_attack(
    fixture_context: dict[str, object],
) -> None:
    jobs = build_forced_jobs(**fixture_context)  # type: ignore[arg-type]

    assert all(
        job.clean_source_chunk
        == _chunk("CH_SYN_GOLD", "Synthetic gold fact.")
        for job in jobs
    )
    assert {
        job.clean_source_commitment for job in jobs
    } == {"45b48cdda0bc39c40f793db3b25544b1411c809c64f9b73c9b0974227288d884"}
    assert {
        job.attack_commitment for job in jobs
    } == {"7abf3efd97cccc1f19950a7f560cb2948f33f35a7af80e4366791774cd25b670"}


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"defense_ids": ()}, "defense_ids must be non-empty"),
        ({"defense_ids": (DefenseId.D0, DefenseId.D0)}, "defense_ids must be unique"),
        ({"malicious_rank": 5}, "malicious_rank must satisfy"),
        (
            {
                "ordered_benign_chunks": (
                    RetrievedChunk(
                        chunk=_chunk("CH_SYN_ALPHA", "Synthetic alpha fact."), score=0.9, rank=1
                    ),
                    RetrievedChunk(
                        chunk=_chunk("CH_SYN_GOLD", "Synthetic gold fact."), score=0.8, rank=3
                    ),
                )
            },
            "contiguous",
        ),
        ({"question": _question(split=Split.TEST)}, "same split"),
        ({"attack": _attack(base_context_id="CTX_SYN_OTHER")}, "base_context_id"),
    ],
)
def test_forced_jobs_validate_contracts_before_building(
    fixture_context: dict[str, object], update: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_forced_jobs(**(fixture_context | update))  # type: ignore[arg-type]


def test_forced_job_ids_include_research_inputs_not_only_defense_position(
    fixture_context: dict[str, object],
) -> None:
    inputs = fixture_context | {"defense_ids": (DefenseId.D0,)}
    baseline = build_forced_jobs(**inputs)  # type: ignore[arg-type]
    changed = build_forced_jobs(
        **(inputs | {"question": _question(question="A changed synthetic question?")})
    )  # type: ignore[arg-type]

    assert baseline[0].job_id != changed[0].job_id


@pytest.mark.parametrize(
    ("question", "message"),
    [
        (
            _question().model_copy(update={"answerable": False, "gold_chunk_ids": ()}),
            "question must be answerable for attack manifests",
        ),
        (
            _question().model_copy(update={"gold_chunk_ids": ("CH_SYN_OTHER",)}),
            "gold_chunk_id must be one of question.gold_chunk_ids",
        ),
    ],
)
def test_forced_jobs_reject_questions_ineligible_for_attack_manifests(
    fixture_context: dict[str, object], question: Question, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        build_forced_jobs(**(fixture_context | {"question": question}))  # type: ignore[arg-type]


def _e2e_context(*, malicious_vector: object = np.asarray([1.0, 0.0])) -> dict[str, object]:
    alpha = _chunk("CH_SYN_ALPHA", "Synthetic alpha fact.")
    gold = _chunk("CH_SYN_GOLD", "Synthetic gold fact.")
    gamma = _chunk("CH_SYN_GAMMA", "Synthetic gamma fact.")
    return {
        "question": _question(),
        "base_context_id": "CTX_SYN_001",
        "corpus_chunks": (alpha, gold, gamma),
        "corpus_matrix": np.asarray([[0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]),
        "query_vector": np.asarray([1.0, 0.0]),
        "malicious_vector": malicious_vector,
        "gold_chunk_id": gold.chunk_id,
        "attack": _attack(),
        "top_k": 3,
        "defense_ids": tuple(DefenseId),
        "config_hash": "CFG_SYN_001",
    }


@pytest.mark.parametrize(
    ("question", "message"),
    [
        (
            _question().model_copy(update={"answerable": False, "gold_chunk_ids": ()}),
            "question must be answerable for attack manifests",
        ),
        (
            _question().model_copy(update={"gold_chunk_ids": ("CH_SYN_OTHER",)}),
            "gold_chunk_id must be one of question.gold_chunk_ids",
        ),
    ],
)
def test_e2e_jobs_reject_questions_ineligible_for_attack_manifests(
    question: Question, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        build_e2e_jobs(**(_e2e_context() | {"question": question}))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("question", "message"),
    [
        (
            _question().model_copy(update={"answerable": False, "gold_chunk_ids": ()}),
            "question must be answerable for attack manifests",
        ),
        (
            _question().model_copy(update={"gold_chunk_ids": ("CH_SYN_OTHER",)}),
            "gold_chunk_id must be one of question.gold_chunk_ids",
        ),
    ],
)
def test_e2e_jobs_reject_ineligible_questions_before_retrieval(
    monkeypatch: pytest.MonkeyPatch, question: Question, message: str
) -> None:
    import thesis_rag.manifests as manifests

    monkeypatch.setattr(
        manifests,
        "exact_cosine_top_k",
        lambda *args, **kwargs: pytest.fail("retrieval must not run for invalid inputs"),
    )

    with pytest.raises(ValueError, match=f"^{message}$"):
        build_e2e_jobs(**(_e2e_context() | {"question": question}))  # type: ignore[arg-type]


def test_e2e_jobs_replace_gold_embedding_retrieve_once_and_share_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import thesis_rag.manifests as manifests

    calls = 0
    exact_retrieval = manifests.exact_cosine_top_k

    def count_retrieval(*args: object, **kwargs: object) -> tuple[RetrievedChunk, ...]:
        nonlocal calls
        calls += 1
        return exact_retrieval(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(manifests, "exact_cosine_top_k", count_retrieval)
    jobs = build_e2e_jobs(**_e2e_context())  # type: ignore[arg-type]
    first = jobs[0]

    assert calls == 1
    assert first.track == "e2e"
    assert first.ordered_chunk_ids[0].startswith("MCH_")
    assert "CH_SYN_GOLD" not in first.ordered_chunk_ids
    assert first.malicious_chunk_rank == 1
    assert all(job.ordered_chunks is first.ordered_chunks for job in jobs)
    assert all(job.attack is first.attack for job in jobs)
    assert all(job.research_inputs is first.research_inputs for job in jobs)
    assert isinstance(first.research_inputs, E2EResearchInputs)
    assert "corpus_matrix" not in first.research_inputs.model_fields_set
    assert "query_vector" not in first.research_inputs.model_fields_set
    assert "malicious_vector" not in first.research_inputs.model_fields_set
    assert len(first.research_inputs.corpus_matrix_commitment) == 64
    assert len(first.research_inputs.query_vector_commitment) == 64
    assert len(first.research_inputs.malicious_vector_commitment) == 64
    assert {
        job.research_input_hash for job in jobs
    } == {canonical_sha256(first.research_inputs)}


def test_e2e_jobs_report_no_malicious_rank_when_not_retrieved() -> None:
    inputs = _e2e_context(malicious_vector=np.asarray([0.0, 1.0])) | {"top_k": 2}

    jobs = build_e2e_jobs(**inputs)  # type: ignore[arg-type]

    assert jobs[0].malicious_chunk_rank is None
    assert all(job.ordered_chunks is jobs[0].ordered_chunks for job in jobs)
    assert all(
        job.clean_source_chunk
        == _chunk("CH_SYN_GOLD", "Synthetic gold fact.")
        for job in jobs
    )
    assert {
        job.clean_source_commitment for job in jobs
    } == {"45b48cdda0bc39c40f793db3b25544b1411c809c64f9b73c9b0974227288d884"}
    assert {
        job.attack_commitment for job in jobs
    } == {"7abf3efd97cccc1f19950a7f560cb2948f33f35a7af80e4366791774cd25b670"}


def test_e2e_jobs_validate_vector_and_shared_contracts_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import thesis_rag.manifests as manifests

    monkeypatch.setattr(
        manifests,
        "exact_cosine_top_k",
        lambda *args, **kwargs: pytest.fail("retrieval must not run for invalid inputs"),
    )

    with pytest.raises(ValueError, match="malicious_vector dimensions must match"):
        build_e2e_jobs(**_e2e_context(malicious_vector=np.asarray([1.0, 0.0, 0.0])))  # type: ignore[arg-type]


@pytest.mark.parametrize("track", ["forced", "e2e"])
def test_research_inputs_reconstruct_the_exact_manifest_job(
    fixture_context: dict[str, object], track: str
) -> None:
    job = (
        build_forced_jobs(
            **(fixture_context | {"defense_ids": (DefenseId.D3,)})
        )[0]  # type: ignore[arg-type]
        if track == "forced"
        else build_e2e_jobs(
            **(_e2e_context() | {"defense_ids": (DefenseId.D3,)})
        )[0]  # type: ignore[arg-type]
    )

    assert reconstruct_job_from_research_inputs(job) == job


def test_e2e_job_recomputes_against_committed_external_vectors() -> None:
    inputs = _e2e_context()
    job = build_e2e_jobs(
        **(inputs | {"defense_ids": (DefenseId.D4,)})
    )[0]  # type: ignore[arg-type]

    assert verify_e2e_job_from_sources(
        job,
        corpus_chunks=inputs["corpus_chunks"],
        corpus_matrix=inputs["corpus_matrix"],
        query_vector=inputs["query_vector"],
        malicious_vector=inputs["malicious_vector"],
    ) == job


def test_e2e_job_rejects_changed_external_vector_commitment() -> None:
    inputs = _e2e_context()
    job = build_e2e_jobs(
        **(inputs | {"defense_ids": (DefenseId.D4,)})
    )[0]  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="malicious vector commitment mismatch"):
        verify_e2e_job_from_sources(
            job,
            corpus_chunks=inputs["corpus_chunks"],
            corpus_matrix=inputs["corpus_matrix"],
            query_vector=inputs["query_vector"],
            malicious_vector=np.asarray([0.0, 1.0]),
        )


def test_clean_jobs_bind_all_defenses_to_one_retrieval_context() -> None:
    question = _question()
    ordered_chunks = (
        RetrievedChunk(
            chunk=_chunk("CH_SYN_ALPHA", "Synthetic alpha fact."),
            score=0.9,
            rank=1,
        ),
        RetrievedChunk(
            chunk=_chunk("CH_SYN_BETA", "Synthetic beta fact."),
            score=0.8,
            rank=2,
        ),
    )

    jobs = build_clean_jobs(
        question=question,
        ordered_chunks=ordered_chunks,
        defense_ids=tuple(DefenseId),
        config_hash="CFG_SYN_001",
    )

    assert len(jobs) == 5
    assert tuple(job.defense_id for job in jobs) == tuple(DefenseId)
    assert len({job.research_input_hash for job in jobs}) == 1
    assert all(job.track == "clean" for job in jobs)
    assert all(job.attack is None for job in jobs)
    assert all(job.ordered_chunks is jobs[0].ordered_chunks for job in jobs)
    assert all(reconstruct_job_from_research_inputs(job) == job for job in jobs)


def test_clean_job_reconstruction_rejects_substituted_context() -> None:
    job = build_clean_jobs(
        question=_question(),
        ordered_chunks=(
            RetrievedChunk(
                chunk=_chunk("CH_SYN_ALPHA", "Synthetic alpha fact."),
                score=0.9,
                rank=1,
            ),
        ),
        defense_ids=(DefenseId.D0,),
        config_hash="CFG_SYN_001",
    )[0]
    forged_chunk = RetrievedChunk(
        chunk=_chunk("CH_FORGED", "Forged context."),
        score=1.0,
        rank=1,
    )
    forged = job.model_copy(
        update={
            "job_id": "JOB_PENDING",
            "ordered_chunks": (forged_chunk,),
            "ordered_chunk_ids": ("CH_FORGED",),
        }
    )
    forged = forged.model_copy(
        update={"job_id": expected_experiment_job_id(forged)}
    )

    with pytest.raises(ValueError, match="reconstructed research manifest"):
        reconstruct_job_from_research_inputs(forged)


def test_research_inputs_reject_a_forged_hidden_e2e_context_and_rebound_job_id() -> None:
    job = build_e2e_jobs(
        **(
            _e2e_context(malicious_vector=np.asarray([0.0, 1.0]))
            | {"top_k": 2, "defense_ids": (DefenseId.D0,)}
        )
    )[0]  # type: ignore[arg-type]
    forged_chunk = RetrievedChunk(
        chunk=_chunk("CH_FORGED_BENIGN", "Arbitrary substituted context."),
        score=1.0,
        rank=1,
    )
    forged = job.model_copy(
        update={
            "job_id": "JOB_PENDING",
            "ordered_chunks": (forged_chunk,),
            "ordered_chunk_ids": (forged_chunk.chunk.chunk_id,),
        }
    )
    forged = forged.model_copy(
        update={"job_id": expected_experiment_job_id(forged)}
    )

    with pytest.raises(ValueError, match="reconstructed research manifest"):
        reconstruct_job_from_research_inputs(forged)
