import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import numpy as np
from pydantic import Field, StrictBool, field_validator, model_validator

from thesis_rag.canonical import JsonValue, canonical_sha256
from thesis_rag.domain import (
    DefenseId,
    ExperimentRecord,
    FiniteFloat,
    FrozenRecord,
    Sha256Hex,
)


class AttackOutcome(FrozenRecord):
    track: Literal["forced", "e2e"]
    base_context_id: str = Field(min_length=1)
    attack_id: str = Field(min_length=1)
    defense_id: DefenseId
    attack_success: StrictBool

    @field_validator("base_context_id", "attack_id")
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("attack outcome identifiers must be nonblank")
        return stripped


class DefenseAttackSummary(FrozenRecord):
    defense_id: DefenseId
    success_count: int = Field(strict=True, ge=0)
    trial_count: int = Field(strict=True, ge=1)
    asr: FiniteFloat

    @model_validator(mode="after")
    def validate_rate(self) -> "DefenseAttackSummary":
        if self.success_count > self.trial_count:
            raise ValueError("attack successes cannot exceed trials")
        expected = self.success_count / self.trial_count
        if not math.isclose(self.asr, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("ASR must match attack counts")
        return self


class AttackOutcomeSummary(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    track: Literal["forced", "e2e"]
    cluster_count: int = Field(strict=True, ge=1)
    attack_count: int = Field(strict=True, ge=1)
    record_count: int = Field(strict=True, ge=1)
    by_defense: tuple[DefenseAttackSummary, ...] = Field(min_length=1)


class ClusterBootstrapResult(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    baseline: DefenseId
    treatment: DefenseId
    cluster_count: int = Field(strict=True, ge=2)
    attack_count: int = Field(strict=True, ge=1)
    baseline_asr: FiniteFloat
    treatment_asr: FiniteFloat
    asr_reduction: FiniteFloat
    confidence_level: FiniteFloat
    ci_lower: FiniteFloat
    ci_upper: FiniteFloat
    iterations: int = Field(strict=True, ge=100)
    seed: int
    distribution_sha256: Sha256Hex


class HolmAdjustment(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    raw_p_values: dict[str, FiniteFloat]
    adjusted_p_values: dict[str, FiniteFloat]


class GeeDefenseComparison(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    baseline: DefenseId
    treatment: DefenseId
    cluster_count: int = Field(strict=True, ge=2)
    attack_count: int = Field(strict=True, ge=1)
    covariance_type: Literal["robust"] = "robust"
    working_correlation: Literal["exchangeable"] = "exchangeable"
    log_odds_ratio: FiniteFloat
    standard_error: FiniteFloat
    odds_ratio: FiniteFloat
    ci_lower: FiniteFloat
    ci_upper: FiniteFloat
    p_value: FiniteFloat


def attack_outcomes_from_experiment_records(
    records: Sequence[ExperimentRecord],
    *,
    expected_track: Literal["forced", "e2e"],
    required_defense_ids: tuple[DefenseId, ...] = tuple(DefenseId),
) -> tuple[AttackOutcome, ...]:
    if not records:
        raise ValueError("attack outcome conversion requires experiment records")
    if (
        not required_defense_ids
        or len(set(required_defense_ids)) != len(required_defense_ids)
    ):
        raise ValueError("required defense IDs must be non-empty and unique")
    if any(record.track != expected_track for record in records):
        raise ValueError("experiment record track mismatch")
    if any(record.technical_failure for record in records):
        raise ValueError(
            "technical failures must be resolved before attack analysis"
        )
    outcomes: list[AttackOutcome] = []
    for record in records:
        if (
            record.base_context_id is None
            or record.attack_id is None
            or record.attack_success is None
        ):
            raise ValueError(
                "attack experiment record is missing a required outcome field"
            )
        outcomes.append(
            AttackOutcome(
                track=expected_track,
                base_context_id=record.base_context_id,
                attack_id=record.attack_id,
                defense_id=record.defense_id,
                attack_success=record.attack_success,
            )
        )
    indexed = _index_outcomes(outcomes)
    required_defenses = set(required_defense_ids)
    attack_ids = {outcome.attack_id for outcome in outcomes}
    for attack_id in attack_ids:
        observed = {
            defense_id
            for indexed_attack_id, defense_id in indexed
            if indexed_attack_id == attack_id
        }
        if observed != required_defenses:
            raise ValueError(
                f"required defense coverage mismatch for {attack_id}"
            )
    return tuple(outcomes)


def summarize_attack_outcomes(
    outcomes: Sequence[AttackOutcome],
) -> AttackOutcomeSummary:
    indexed = _index_outcomes(outcomes)
    _require_shared_defense_set(indexed)
    observed_defenses = {
        outcome.defense_id for outcome in indexed.values()
    }
    summaries = tuple(
        DefenseAttackSummary(
            defense_id=defense_id,
            success_count=sum(
                outcome.attack_success
                for outcome in indexed.values()
                if outcome.defense_id is defense_id
            ),
            trial_count=sum(
                outcome.defense_id is defense_id
                for outcome in indexed.values()
            ),
            asr=(
                sum(
                    outcome.attack_success
                    for outcome in indexed.values()
                    if outcome.defense_id is defense_id
                )
                / sum(
                    outcome.defense_id is defense_id
                    for outcome in indexed.values()
                )
            ),
        )
        for defense_id in DefenseId
        if defense_id in observed_defenses
    )
    first = next(iter(indexed.values()))
    return AttackOutcomeSummary(
        track=first.track,
        cluster_count=len(
            {outcome.base_context_id for outcome in indexed.values()}
        ),
        attack_count=len({outcome.attack_id for outcome in indexed.values()}),
        record_count=len(indexed),
        by_defense=summaries,
    )


def cluster_bootstrap_asr_reduction(
    outcomes: Sequence[AttackOutcome],
    *,
    baseline: DefenseId,
    treatment: DefenseId,
    iterations: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> ClusterBootstrapResult:
    if iterations < 100:
        raise ValueError("cluster bootstrap requires at least 100 iterations")
    if (
        isinstance(confidence_level, bool)
        or not math.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence level must be finite and in (0, 1)")
    paired = _paired_outcomes(
        outcomes,
        baseline=baseline,
        treatment=treatment,
    )
    context_ids = tuple(
        sorted({outcome.base_context_id for pair in paired for outcome in pair})
    )
    if len(context_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two base contexts")

    pairs_by_context: dict[
        str,
        list[tuple[AttackOutcome, AttackOutcome]],
    ] = {context_id: [] for context_id in context_ids}
    for baseline_row, treatment_row in paired:
        pairs_by_context[baseline_row.base_context_id].append(
            (baseline_row, treatment_row)
        )
    baseline_successes = np.asarray(
        [
            sum(row[0].attack_success for row in pairs_by_context[context_id])
            for context_id in context_ids
        ],
        dtype=np.float64,
    )
    treatment_successes = np.asarray(
        [
            sum(row[1].attack_success for row in pairs_by_context[context_id])
            for context_id in context_ids
        ],
        dtype=np.float64,
    )
    trial_counts = np.asarray(
        [len(pairs_by_context[context_id]) for context_id in context_ids],
        dtype=np.float64,
    )

    baseline_asr = float(baseline_successes.sum() / trial_counts.sum())
    treatment_asr = float(treatment_successes.sum() / trial_counts.sum())
    reduction = baseline_asr - treatment_asr
    random = np.random.default_rng(seed)
    sampled_indices = random.integers(
        0,
        len(context_ids),
        size=(iterations, len(context_ids)),
    )
    sampled_trials = trial_counts[sampled_indices].sum(axis=1)
    distribution = (
        baseline_successes[sampled_indices].sum(axis=1) / sampled_trials
        - treatment_successes[sampled_indices].sum(axis=1) / sampled_trials
    )
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower, ci_upper = np.quantile(
        distribution,
        (alpha, 1.0 - alpha),
        method="linear",
    )
    distribution_values = tuple(float(value) for value in distribution)
    return ClusterBootstrapResult(
        baseline=baseline,
        treatment=treatment,
        cluster_count=len(context_ids),
        attack_count=len(paired),
        baseline_asr=baseline_asr,
        treatment_asr=treatment_asr,
        asr_reduction=reduction,
        confidence_level=confidence_level,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        iterations=iterations,
        seed=seed,
        distribution_sha256=canonical_sha256(
            cast(JsonValue, distribution_values)
        ),
    )


def holm_adjust(p_values: Mapping[str, float]) -> HolmAdjustment:
    if not p_values:
        raise ValueError("Holm adjustment requires at least one comparison")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in p_values.items():
        name = raw_name.strip()
        if not name:
            raise ValueError("Holm comparison names must be nonblank")
        value = float(raw_value)
        if (
            isinstance(raw_value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("p-values must be finite and in [0, 1]")
        if name in normalized:
            raise ValueError("Holm comparison names must be unique")
        normalized[name] = value

    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    adjusted_sorted: dict[str, float] = {}
    running_maximum = 0.0
    comparison_count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        adjusted = min(1.0, (comparison_count - index) * value)
        running_maximum = max(running_maximum, adjusted)
        adjusted_sorted[name] = running_maximum
    return HolmAdjustment(
        raw_p_values=dict(normalized),
        adjusted_p_values={
            name: adjusted_sorted[name] for name in normalized
        },
    )


def fit_paired_defense_gee(
    outcomes: Sequence[AttackOutcome],
    *,
    baseline: DefenseId,
    treatment: DefenseId,
    confidence_level: float = 0.95,
) -> GeeDefenseComparison:
    if (
        isinstance(confidence_level, bool)
        or not math.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence level must be finite and in (0, 1)")
    paired = _paired_outcomes(
        outcomes,
        baseline=baseline,
        treatment=treatment,
    )
    context_ids = tuple(
        sorted({row.base_context_id for pair in paired for row in pair})
    )
    if len(context_ids) < 4:
        raise ValueError("GEE requires at least four base contexts")
    ordered_rows = tuple(
        row
        for baseline_row, treatment_row in paired
        for row in (baseline_row, treatment_row)
    )
    response = np.asarray(
        [float(row.attack_success) for row in ordered_rows],
        dtype=np.float64,
    )
    if len(set(response.tolist())) != 2:
        raise ValueError("GEE requires both success and failure outcomes")
    design = np.asarray(
        [
            (1.0, float(row.defense_id is treatment))
            for row in ordered_rows
        ],
        dtype=np.float64,
    )
    context_index = {
        context_id: index for index, context_id in enumerate(context_ids)
    }
    groups = np.asarray(
        [context_index[row.base_context_id] for row in ordered_rows],
        dtype=np.int64,
    )

    try:
        from statsmodels.genmod.cov_struct import Exchangeable  # type: ignore[import-untyped]
        from statsmodels.genmod.families import Binomial  # type: ignore[import-untyped]
        from statsmodels.genmod.generalized_estimating_equations import (  # type: ignore[import-untyped]
            GEE,
        )
    except ImportError as error:
        raise RuntimeError(
            "GEE analysis requires the project analysis dependencies"
        ) from error

    fitted = GEE(
        response,
        design,
        groups=groups,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit(cov_type="robust")
    if not bool(fitted.converged):
        raise ValueError("GEE failed to converge")
    coefficient = float(fitted.params[1])
    standard_error = float(fitted.bse[1])
    p_value = float(fitted.pvalues[1])
    confidence_interval = np.asarray(
        fitted.conf_int(alpha=1.0 - confidence_level),
        dtype=np.float64,
    )[1]
    values = (
        coefficient,
        standard_error,
        p_value,
        float(confidence_interval[0]),
        float(confidence_interval[1]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("GEE produced non-finite estimates")
    return GeeDefenseComparison(
        baseline=baseline,
        treatment=treatment,
        cluster_count=len(context_ids),
        attack_count=len(paired),
        log_odds_ratio=coefficient,
        standard_error=standard_error,
        odds_ratio=math.exp(coefficient),
        ci_lower=math.exp(float(confidence_interval[0])),
        ci_upper=math.exp(float(confidence_interval[1])),
        p_value=p_value,
    )


def _index_outcomes(
    outcomes: Sequence[AttackOutcome],
) -> dict[tuple[str, DefenseId], AttackOutcome]:
    if not outcomes:
        raise ValueError("attack analysis requires at least one outcome")
    indexed: dict[tuple[str, DefenseId], AttackOutcome] = {}
    attack_contexts: dict[str, str] = {}
    tracks = {outcome.track for outcome in outcomes}
    if len(tracks) != 1:
        raise ValueError("attack analysis outcomes must use one track")
    for outcome in outcomes:
        key = (outcome.attack_id, outcome.defense_id)
        if key in indexed:
            raise ValueError(
                "duplicate attack-defense outcome: "
                f"{outcome.attack_id}/{outcome.defense_id.value}"
            )
        previous_context = attack_contexts.setdefault(
            outcome.attack_id,
            outcome.base_context_id,
        )
        if previous_context != outcome.base_context_id:
            raise ValueError(
                f"attack base-context mismatch: {outcome.attack_id}"
            )
        indexed[key] = outcome

    return indexed


def _paired_outcomes(
    outcomes: Sequence[AttackOutcome],
    *,
    baseline: DefenseId,
    treatment: DefenseId,
) -> tuple[tuple[AttackOutcome, AttackOutcome], ...]:
    if baseline is treatment:
        raise ValueError("baseline and treatment defenses must differ")
    indexed = _index_outcomes(outcomes)
    attack_ids = tuple(sorted({attack_id for attack_id, _ in indexed}))
    paired: list[tuple[AttackOutcome, AttackOutcome]] = []
    for attack_id in attack_ids:
        missing = tuple(
            defense
            for defense in (baseline, treatment)
            if (attack_id, defense) not in indexed
        )
        if missing:
            raise ValueError(
                "paired defense coverage mismatch for "
                f"{attack_id}: missing {','.join(item.value for item in missing)}"
            )
        paired.append(
            (
                indexed[(attack_id, baseline)],
                indexed[(attack_id, treatment)],
            )
        )
    return tuple(paired)


def _require_shared_defense_set(
    indexed: Mapping[tuple[str, DefenseId], AttackOutcome],
) -> None:
    defense_sets: dict[str, set[DefenseId]] = {}
    for attack_id, defense_id in indexed:
        defense_sets.setdefault(attack_id, set()).add(defense_id)
    observed_sets = {
        frozenset(defenses) for defenses in defense_sets.values()
    }
    if len(observed_sets) != 1:
        raise ValueError("attack outcomes do not share one defense set")
