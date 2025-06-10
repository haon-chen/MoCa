import logging
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Dict, List, Optional, cast
import json

import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import set_seed
import os

from vidore_benchmark.evaluation.interfaces import MetadataModel, ViDoReBenchmarkResults
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import ViDoReEvaluatorBEIR
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import load_vision_retriever_from_registry
from vidore_benchmark.utils.data_utils import get_datasets_from_collection
from vidore_benchmark.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

load_dotenv(override=True)
set_seed(42)

app = typer.Typer(
    help="""
    CLI for evaluating vision retrievers.
    Can be used to evaluate on the ViDoRe benchmark and to generate metrics for the ViDoRe leaderboard.
    """,
    no_args_is_help=True,
)


def _sanitize_model_id(
    model_class: str,
    model_name: Optional[str] = None,
) -> str:
    """
    Return sanitized model ID for properly saving metrics as files.
    """
    model_id = model_class
    if model_name:
        model_id += f"_{model_name}"
    model_id = model_id.replace("/", "_")
    return model_id


def _get_metrics_from_vidore_evaluator(
    vision_retriever: BaseVisionRetriever,
    dataset_name: str,
    dataset_format: str,
    split: str,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int] = None,
    dataloader_prebatch_query: Optional[int] = None,
    dataloader_prebatch_passage: Optional[int] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Rooter function to get metrics from the ViDoRe evaluator depending on the dataset format.
    """
    if dataset_format.lower() == "qa":
        ds = cast(Dataset, load_dataset(dataset_name, split=split))
        vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)
    elif dataset_format.lower() == "beir":
        ds = {
            "corpus": cast(Dataset, load_dataset(dataset_name, name="corpus", split=split)),
            "queries": cast(Dataset, load_dataset(dataset_name, name="queries", split=split)),
            "qrels": cast(Dataset, load_dataset(dataset_name, name="qrels", split=split)),
        }
        vidore_evaluator = ViDoReEvaluatorBEIR(vision_retriever)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    metrics = {
        dataset_name: vidore_evaluator.evaluate_dataset(
            ds=ds,
            ds_format=dataset_format,
            batch_query=batch_query,
            batch_passage=batch_passage,
            batch_score=batch_score,
            dataloader_prebatch_query=dataloader_prebatch_query,
            dataloader_prebatch_passage=dataloader_prebatch_passage,
        )
    }
    return metrics


def _load_existing_results(savedir_datasets: Path, dataset_name: str) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """
    Load existing evaluation results if they exist.
    """
    sanitized_dataset_name = dataset_name.replace("/", "_")
    savepath_results = savedir_datasets / f"{sanitized_dataset_name}_metrics.json"
    if savepath_results.exists():
        with open(str(savepath_results), "r", encoding="utf-8") as f:
            results = json.load(f)
            return {dataset_name: results["metrics"][dataset_name]}
    return None


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


@app.command()
def evaluate_retriever(
    model_class: Annotated[str, typer.Option(help="Model class")],
    dataset_format: Annotated[
        str,
        typer.Option(
            help='Dataset format to use for evaluation ("qa" or "beir"). Use "qa" (uses query deduplication) for '
            'ViDoRe Benchmark v1 and "beir" (without query dedup) for ViDoRe Benchmark v2 (not released yet).'
        ),
    ],
    model_name: Annotated[
        Optional[str],
        typer.Option(
            "--model-name",
            help="For Hf transformers-based models, this value is passed to the `model.from_pretrained` method.",
        ),
    ] = None,
    dataset_name: Annotated[Optional[str], typer.Option(help="Hf Hub dataset name.")] = None,
    collection_name: Annotated[
        Optional[str],
        typer.Option(help="Dataset collection to use for evaluation. Can be a Hf collection id or a local dirpath."),
    ] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    batch_query: Annotated[int, typer.Option(help="Batch size for query embedding inference")] = 4,
    batch_passage: Annotated[int, typer.Option(help="Batch size for passages embedding inference")] = 4,
    batch_score: Annotated[Optional[int], typer.Option(help="Batch size for retrieval score computation")] = 4,
    dataloader_prebatch_query: Annotated[
        Optional[int], typer.Option(help="Dataloader prebatch size for queries")
    ] = None,
    dataloader_prebatch_passage: Annotated[
        Optional[int], typer.Option(help="Dataloader prebatch size for passages")
    ] = None,
    num_workers: Annotated[
        int, typer.Option(help="Number of workers for dataloader in retrievers, when supported")
    ] = 0,
    output_dir: Annotated[str, typer.Option(help="Directory where to save the metrics")] = "outputs",
    eval_first_dataset_only: Annotated[
        bool, typer.Option(help="Whether to evaluate only the first dataset in the collection")
    ] = False,
):
    """
    Evaluate a retriever on a given dataset or dataset collection.
    The MTEB retrieval metrics are saved to a JSON file.
    """

    if dataset_name is None and collection_name is None:
        raise ValueError("Please provide a dataset name or collection name")
    elif dataset_name is not None and collection_name is not None:
        raise ValueError("Please provide only one of dataset name or collection name")

    retriever = load_vision_retriever_from_registry(
        model_class,
        pretrained_model_name_or_path=model_name,
        num_workers=num_workers,
    )
    model_id = _sanitize_model_id(model_class, model_name=model_name)

    dataset_names: List[str] = []
    if dataset_name is not None:
        dataset_names = [dataset_name]
    elif collection_name is not None:
        dataset_names = get_datasets_from_collection(collection_name)

    dataset_version = "v2" if dataset_format.lower() == "beir" else "v1"
    output_dir = os.path.join(output_dir, dataset_version)
    savedir_root = Path(output_dir)
    savedir_datasets = savedir_root / model_id.replace("/", "_")
    savedir_datasets.mkdir(parents=True, exist_ok=True)

    metrics_all = {}
    results_all = []

    for dataset_name in tqdm(dataset_names, desc="Processing dataset(s)"):
        if eval_first_dataset_only and dataset_name != dataset_names[0]:
            continue
            
        # Check if results already exist
        sanitized_dataset_name = dataset_name.replace("/", "_")
        result_path = savedir_datasets / f"{sanitized_dataset_name}_metrics.json"
        
        if result_path.exists():
            # Load existing results
            with open(result_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                metrics = {dataset_name: result_data["metrics"][dataset_name]}
                print(f"\n---------------------------\n{dataset_name} (loaded from cache)")
        else:
            # Evaluate dataset
            print(f"\n---------------------------\n{dataset_name}")
            metrics = _get_metrics_from_vidore_evaluator(
                vision_retriever=retriever,
                dataset_name=dataset_name,
                dataset_format=dataset_format,
                split=split,
                batch_query=batch_query,
                batch_passage=batch_passage,
                batch_score=batch_score,
                dataloader_prebatch_query=dataloader_prebatch_query,
                dataloader_prebatch_passage=dataloader_prebatch_passage,
            )
            
            # Save results
            results = ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={dataset_name: metrics[dataset_name]},
            )
            
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=4))
            
            results_all.append(results)
            
        # Print individual dataset score
        print(f"nDCG@5 on {dataset_name}: {metrics[dataset_name]['ndcg_at_5']}")
        metrics_all.update(metrics)

    # Calculate and print average nDCG@5 across all datasets
    ndcg5_scores = [metrics_all[ds_name]['ndcg_at_5'] for ds_name in metrics_all]
    avg_ndcg5 = sum(ndcg5_scores) * 100 / len(ndcg5_scores)
    print(f"\nAverage nDCG@5 across all datasets: {avg_ndcg5:.1f}")

    # Merge all results and save
    if not results_all:
        # If all results were loaded from cache, reconstruct results_all
        for ds_name in metrics_all:
            results_all.append(ViDoReBenchmarkResults(
                metadata=MetadataModel(
                    timestamp=datetime.now(),
                    vidore_benchmark_version=version("vidore_benchmark"),
                ),
                metrics={ds_name: metrics_all[ds_name]},
            ))
            
    results_merged = ViDoReBenchmarkResults.merge(results_all)
    savepath_results_merged = savedir_root / f"{model_id}_metrics.json"

    with open(str(savepath_results_merged), "w", encoding="utf-8") as f:
        f.write(results_merged.model_dump_json(indent=4))

    print(f"ViDoRe Benchmark results saved to `{savepath_results_merged}`")


if __name__ == "__main__":
    app()
