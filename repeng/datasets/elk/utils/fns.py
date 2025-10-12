from typing import Callable

from tqdm import tqdm

from repeng.datasets.elk.arc import get_arc
from repeng.datasets.elk.common_sense_qa import get_common_sense_qa
from repeng.datasets.elk.dlk import get_dlk_dataset
from repeng.datasets.elk.geometry_of_truth import get_geometry_of_truth
from repeng.datasets.elk.open_book_qa import get_open_book_qa
from repeng.datasets.elk.race import get_race
from repeng.datasets.elk.true_false import get_true_false_dataset
from repeng.datasets.elk.truthful_model_written import get_truthful_model_written
from repeng.datasets.elk.truthful_qa import get_truthful_qa
from repeng.datasets.elk.types import BinaryRow, DatasetId

_DATASET_FNS: dict[DatasetId, Callable[[], dict[str, BinaryRow]]] = {
    "got_cities": lambda: get_geometry_of_truth("cities"),
    "got_sp_en_trans": lambda: get_geometry_of_truth("sp_en_trans"),
    "got_larger_than": lambda: get_geometry_of_truth("larger_than"),
    "got_cities_cities_conj": lambda: get_geometry_of_truth("cities_cities_conj"),
    "got_cities_cities_disj": lambda: get_geometry_of_truth("cities_cities_disj"),
    "arc_challenge": lambda: get_arc("challenge", "repe"),
    "arc_easy": lambda: get_arc("easy", "repe"),
    "common_sense_qa": lambda: get_common_sense_qa("repe"),
    "open_book_qa": lambda: get_open_book_qa("repe"),
    "race": lambda: get_race("repe"),
    "arc_challenge/simple": lambda: get_arc("challenge", "simple"),
    "arc_easy/simple": lambda: get_arc("easy", "simple"),
    "common_sense_qa/simple": lambda: get_common_sense_qa("simple"),
    "open_book_qa/simple": lambda: get_open_book_qa("simple"),
    "race/simple": lambda: get_race("simple"),
    "truthful_qa": lambda: get_truthful_qa(),
    "truthful_model_written": lambda: get_truthful_model_written(),
    "true_false": get_true_false_dataset,
    "imdb": lambda: get_dlk_dataset("imdb"),
    "imdb/simple": lambda: get_dlk_dataset("imdb/simple"),
    "amazon_polarity": lambda: get_dlk_dataset("amazon_polarity"),
    "ag_news": lambda: get_dlk_dataset("ag_news"),
    "dbpedia_14": lambda: get_dlk_dataset("dbpedia_14"),
    "rte": lambda: get_dlk_dataset("rte"),
    "copa": lambda: get_dlk_dataset("copa"),
    "boolq": lambda: get_dlk_dataset("boolq"),
    "boolq/simple": lambda: get_dlk_dataset("boolq/simple"),
    "piqa": lambda: get_dlk_dataset("piqa"),
}


def get_dataset(dataset_id: DatasetId) -> dict[str, BinaryRow]:
    return _DATASET_FNS[dataset_id]()


def get_datasets(dataset_ids: list[DatasetId]) -> dict[str, BinaryRow]:
    result = {}
    pbar = tqdm(dataset_ids, desc="loading datasets")
    for dataset_id in pbar:
        pbar.set_postfix(dataset=dataset_id)
        result.update(get_dataset(dataset_id))
    return result
