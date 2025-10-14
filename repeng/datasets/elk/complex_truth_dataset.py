"""
Complex Truth Dataset Integration

Loads the complex truth dataset into the standard BinaryRow format
for use in probe training and evaluation experiments.
"""

from repeng.datasets.complex_truth import get_complex_truth_dataset
from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_to_all


def get_complex_truth() -> dict[str, BinaryRow]:
    """
    Returns the complex truth dataset in BinaryRow format.

    Returns:
        Dict mapping unique keys to BinaryRow objects
    """
    result = {}
    statements = get_complex_truth_dataset()

    # Shuffle deterministically for reproducibility
    indices = deterministic_shuffle(list(range(len(statements))), key=str)

    for idx in indices:
        text, label = statements[idx]
        key = f"complex_truth-{idx}"

        result[key] = BinaryRow(
            dataset_id="complex_truth",
            split=split_to_all("complex_truth", str(idx)),
            text=text,
            label=label,
            format_args=dict(),
            group_id=None,  # No paired questions in this dataset
        )

    return result
