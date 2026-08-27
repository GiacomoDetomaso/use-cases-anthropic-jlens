import pytest

from dataset_agent.models.dataset_generation_io_models import QualityAssessmentModel


@pytest.mark.parametrize(
    ("original_intent_preserved", "attack_matches_pattern", "expected_accepted"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_quality_assessment_requires_both_validation_gates(
    original_intent_preserved: bool,
    attack_matches_pattern: bool,
    expected_accepted: bool,
) -> None:
    assessment = QualityAssessmentModel(
        original_intent_preserved=original_intent_preserved,
        attack_present=attack_matches_pattern,
        feedback="Validation completed.",
    )

    assert assessment.accepted is expected_accepted