"""
Additional datasets from the Geometry of Truth paper.

These datasets are loaded from cached CSV files that were downloaded
from the paper's repository.
"""

from pathlib import Path
from typing import cast

import pandas as pd

from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_to_all

# All datasets are now stored locally in the repo
_DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "datasets"


def get_likely() -> dict[str, BinaryRow]:
    """
    Load the 'likely' dataset - nonfactual text with likely/unlikely completions.

    This dataset tests whether probes are merely learning "probable vs improbable text"
    rather than actual truth. From the paper Section 5.3:
    "Probes trained on likely perform poorly... indicating that LLMs linearly represent
    truth-relevant information beyond the plausibility of text."

    Returns:
        Dict mapping unique keys to BinaryRow objects
    """
    csv_path = _DATASETS_DIR / "likely.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"likely.csv not found at {csv_path}. "
            f"Expected to find datasets in: {_DATASETS_DIR}"
        )

    df = pd.read_csv(csv_path)
    result = {}

    for index in deterministic_shuffle(list(range(len(df))), key=str):
        row = df.iloc[index]
        result[f"likely-{index}"] = BinaryRow(
            dataset_id="likely",
            split=split_to_all("likely", str(index)),
            text=cast(str, row["statement"]),
            label=bool(row["label"] == 1),
            format_args=dict(),
            group_id=None,
        )

    return result


def get_counterfact_true_false() -> dict[str, BinaryRow]:
    """
    Load the 'counterfact_true_false' dataset - factual recall statements.

    From Meng et al. 2022 ROME paper. Contains 31,964 factual statements about
    entities and their attributes. This is the largest dataset used in the paper.

    Returns:
        Dict mapping unique keys to BinaryRow objects
    """
    csv_path = _DATASETS_DIR / "counterfact_true_false.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"counterfact_true_false.csv not found at {csv_path}. "
            f"Expected to find datasets in: {_DATASETS_DIR}"
        )

    df = pd.read_csv(csv_path)
    result = {}

    for index in deterministic_shuffle(list(range(len(df))), key=str):
        row = df.iloc[index]
        result[f"counterfact_true_false-{index}"] = BinaryRow(
            dataset_id="counterfact_true_false",
            split=split_to_all("counterfact_true_false", str(index)),
            text=cast(str, row["statement"]),
            label=bool(row["label"] == 1),
            format_args=dict(),
            group_id=None,
        )

    return result
