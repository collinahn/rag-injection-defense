from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher

from pydantic import Field

from thesis_rag.attack_dataset import (
    TestAttackFactorialSummary,
    validate_test_attack_factorial,
)
from thesis_rag.detector_pilot import (
    HardNegativeCandidate,
    TestHardNegativeDatasetSummary,
    validate_hard_negative_split_integrity,
    validate_test_hard_negative_dataset,
)
from thesis_rag.domain import (
    AttackCase,
    AttackForm,
    AttackGoal,
    AttackPosition,
    FiniteFloat,
    FrozenRecord,
    Language,
    Split,
    normalize_research_text,
)
from thesis_rag.evaluation import validate_split_integrity
from thesis_rag.qa import QuestionType, ResearchQuestion


class QuestionSimilarityPair(FrozenRecord):
    development_question_id: str = Field(min_length=1)
    test_question_id: str = Field(min_length=1)
    similarity: FiniteFloat


class ResearchDataIntegrityReport(FrozenRecord):
    schema_version: str
    development_question_count: int = Field(strict=True, ge=0)
    test_question_count: int = Field(strict=True, ge=0)
    development_attack_count: int = Field(strict=True, ge=0)
    test_attack_count: int = Field(strict=True, ge=0)
    development_hard_negative_count: int = Field(strict=True, ge=0)
    test_hard_negative_count: int = Field(strict=True, ge=0)
    exact_question_overlap_count: int = Field(strict=True, ge=0)
    exact_hard_negative_overlap_count: int = Field(strict=True, ge=0)
    test_attack_summary: TestAttackFactorialSummary
    test_hard_negative_summary: TestHardNegativeDatasetSummary
    near_duplicate_threshold: FiniteFloat
    near_duplicate_question_pairs: tuple[QuestionSimilarityPair, ...]


def find_near_duplicate_question_pairs(
    development_questions: Sequence[ResearchQuestion],
    test_questions: Sequence[ResearchQuestion],
    *,
    threshold: float = 0.65,
) -> tuple[QuestionSimilarityPair, ...]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("question similarity threshold must be in [0, 1]")
    pairs: list[QuestionSimilarityPair] = []
    for development_question in development_questions:
        development_text = normalize_research_text(development_question.question)
        for test_question in test_questions:
            test_text = normalize_research_text(test_question.question)
            similarity = SequenceMatcher(
                None,
                development_text,
                test_text,
                autojunk=False,
            ).ratio()
            if similarity < threshold:
                continue
            pairs.append(
                QuestionSimilarityPair(
                    development_question_id=development_question.question_id,
                    test_question_id=test_question.question_id,
                    similarity=similarity,
                )
            )
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (
                -pair.similarity,
                pair.development_question_id,
                pair.test_question_id,
            ),
        )
    )


def validate_research_data_splits(
    development_questions: Sequence[ResearchQuestion],
    test_questions: Sequence[ResearchQuestion],
    development_attacks: Sequence[AttackCase],
    test_attacks: Sequence[AttackCase],
    development_hard_negatives: Sequence[HardNegativeCandidate],
    test_hard_negatives: Sequence[HardNegativeCandidate],
    *,
    near_duplicate_threshold: float = 0.65,
) -> ResearchDataIntegrityReport:
    _require_count(development_questions, 30, "development questions")
    _require_count(test_questions, 100, "test questions")
    _require_count(development_attacks, 60, "development attacks")
    _require_count(test_attacks, 180, "test attacks")
    _require_count(
        development_hard_negatives,
        60,
        "development hard negatives",
    )
    _require_count(test_hard_negatives, 100, "test hard negatives")
    _require_split(development_questions, Split.DEV, "development questions")
    _require_split(test_questions, Split.TEST, "test questions")
    _require_split(development_attacks, Split.DEV, "development attacks")
    _require_split(test_attacks, Split.TEST, "test attacks")

    validate_split_integrity(
        tuple(development_questions) + tuple(test_questions),
        tuple(development_attacks) + tuple(test_attacks),
    )
    _validate_development_attack_factorial(development_attacks)
    test_attack_summary = validate_test_attack_factorial(
        test_attacks,
        test_questions,
    )
    _validate_test_attack_question_type_balance(test_attacks, test_questions)

    validate_hard_negative_split_integrity(
        development_hard_negatives,
        test_hard_negatives,
    )
    test_hard_negative_summary = validate_test_hard_negative_dataset(test_hard_negatives)

    near_duplicates = find_near_duplicate_question_pairs(
        development_questions,
        test_questions,
        threshold=near_duplicate_threshold,
    )
    return ResearchDataIntegrityReport(
        schema_version="1.0.0",
        development_question_count=len(development_questions),
        test_question_count=len(test_questions),
        development_attack_count=len(development_attacks),
        test_attack_count=len(test_attacks),
        development_hard_negative_count=len(development_hard_negatives),
        test_hard_negative_count=len(test_hard_negatives),
        exact_question_overlap_count=0,
        exact_hard_negative_overlap_count=0,
        test_attack_summary=test_attack_summary,
        test_hard_negative_summary=test_hard_negative_summary,
        near_duplicate_threshold=near_duplicate_threshold,
        near_duplicate_question_pairs=near_duplicates,
    )


def _validate_development_attack_factorial(
    attacks: Sequence[AttackCase],
) -> None:
    cell_counts = Counter(
        (
            attack.goal,
            attack.language,
            attack.form,
            attack.position,
        )
        for attack in attacks
    )
    expected = Counter(
        {
            (goal, language, form, position): 1
            for goal in AttackGoal
            for language in Language
            for form in AttackForm
            for position in AttackPosition
        }
    )
    if cell_counts != expected:
        raise ValueError("development attack factorial cells are not balanced")
    base_counts = Counter(attack.base_context_id for attack in attacks)
    if len(base_counts) != 20 or set(base_counts.values()) != {3}:
        raise ValueError("development attack factorial requires 20 contexts with 3 attacks each")


def _validate_test_attack_question_type_balance(
    attacks: Sequence[AttackCase],
    questions: Sequence[ResearchQuestion],
) -> None:
    questions_by_id = {question.question_id: question for question in questions}
    base_context_ids = {attack.base_context_id for attack in attacks}
    type_counts = Counter(
        questions_by_id[context_id].question_type for context_id in base_context_ids
    )
    expected = Counter(
        {
            QuestionType.SINGLE_SOURCE: 51,
            QuestionType.MULTI_SOURCE: 9,
        }
    )
    if type_counts != expected:
        raise ValueError(
            "test attack contexts must contain 51 single-source and 9 multi-source questions"
        )


def _require_count(
    records: Sequence[object],
    expected: int,
    label: str,
) -> None:
    if len(records) != expected:
        raise ValueError(f"{label} must contain exactly {expected} records")


def _require_split(
    records: Sequence[object],
    expected: Split,
    label: str,
) -> None:
    if any(getattr(record, "split", None) is not expected for record in records):
        raise ValueError(f"{label} must use {expected.value} split")
