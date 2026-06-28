import argparse
import gc
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import torch
from scipy.ndimage import find_objects, label
from torch.utils.data import Dataset
from tqdm import tqdm

from algorithms import get_algorithm, list_algorithms
from datasets import CHiMeNoiseDataset, get_pitch_dataset, list_pitch_datasets


def evaluate_voicing_detection(
    pred_voiced: np.ndarray, true_voiced: np.ndarray
) -> Dict:
    true_pos = np.sum(pred_voiced & true_voiced)
    false_pos = np.sum(pred_voiced & ~true_voiced)
    false_neg = np.sum(~pred_voiced & true_voiced)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_pitch_accuracy(
    pitch_pred: np.ndarray,
    pitch_true: np.ndarray,
    valid_mask: np.ndarray,
    epsilon: float = 50.0,
    gross_error_threshold: float = 200.0,
) -> Dict:
    if len(pitch_pred) != len(pitch_true):
        raise ValueError(
            f"Length mismatch: pred={len(pitch_pred)}, true={len(pitch_true)}"
        )

    if not np.any(valid_mask):
        return {
            "rmse": np.nan,
            "cents_error": np.nan,
            "rpa": np.nan,
            "rca": np.nan,
            "octave_error_rate": np.nan,
            "gross_error_rate": np.nan,
            "valid_frames": 0,
        }

    pred = pitch_pred[valid_mask]
    true = pitch_true[valid_mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        cents_diff = np.abs(1200 * np.log2(pred / true))

    rpa = np.nanmean(cents_diff < epsilon)
    wrapped_cents_diff = cents_diff % 1200
    chroma_diff = np.minimum(wrapped_cents_diff, 1200 - wrapped_cents_diff)
    rca = np.nanmean(chroma_diff < epsilon)
    gross_error_rate = np.nanmean(cents_diff > gross_error_threshold)

    relative_error = np.abs(pred - true) / (true + np.finfo(float).eps)
    octave_errors = np.logical_or(
        relative_error > 0.4, (cents_diff > 1100) & (cents_diff < 1300)
    )
    octave_error_rate = np.nanmean(octave_errors)

    return {
        "rmse": np.sqrt(np.nanmean((pred - true) ** 2)),
        "cents_error": np.nanmean(cents_diff),
        "rpa": rpa,
        "rca": rca,
        "octave_error_rate": octave_error_rate,
        "gross_error_rate": gross_error_rate,
        "valid_frames": int(np.sum(valid_mask)),
    }


def evaluate_pitch_smoothness(
    pitch_pred: np.ndarray, pred_voicing: np.ndarray, true_voicing: np.ndarray
) -> Dict[str, float]:
    relative_smoothness, continuity_breaks = np.nan, np.nan
    voiced_idx = np.where(pred_voicing)[0]
    if len(voiced_idx) >= 2:
        consecutive_mask = np.diff(voiced_idx) == 1
        starts_idx = voiced_idx[:-1][consecutive_mask]
        ends_idx = voiced_idx[1:][consecutive_mask]
        if starts_idx.size > 0:
            pitch_starts = pitch_pred[starts_idx]
            pitch_ends = pitch_pred[ends_idx]
            valid_pairs_mask = (pitch_starts > 0) & (pitch_ends > 0)
            if np.any(valid_pairs_mask):
                pitch_starts = pitch_starts[valid_pairs_mask]
                pitch_ends = pitch_ends[valid_pairs_mask]
                rel_changes = np.abs(pitch_ends - pitch_starts) / (pitch_starts + 1e-8)
                mean_chg, std_chg = np.mean(rel_changes), np.std(rel_changes)
                if mean_chg > 1e-9:
                    relative_smoothness = std_chg / mean_chg
                else:
                    relative_smoothness = 0.0 if std_chg < 1e-8 else np.nan

    labeled_segments, num_segments = label(true_voicing)
    if num_segments > 0:
        gt_segments = find_objects(labeled_segments)
        break_count = 0
        total_relevant_segments = 0
        for seg_slice_tuple in gt_segments:
            seg_slice = seg_slice_tuple[0]
            if seg_slice.stop - seg_slice.start > 1:
                total_relevant_segments += 1
                if not np.all(pred_voicing[seg_slice]):
                    break_count += 1
        if total_relevant_segments > 0:
            continuity_breaks = break_count / total_relevant_segments

    return {
        "relative_smoothness": float(relative_smoothness),
        "continuity_breaks": float(continuity_breaks),
    }


def calculate_processed_metrics(voicing_metrics: Dict, pitch_metrics: Dict) -> Dict:
    cents_error = pitch_metrics.get("cents_error", 500)
    if cents_error is None:
        cents_error = 500

    octave_error_rate = pitch_metrics.get("octave_error_rate", 1.0)
    if octave_error_rate is None:
        octave_error_rate = 1.0

    gross_error_rate = pitch_metrics.get("gross_error_rate", 1.0)
    if gross_error_rate is None:
        gross_error_rate = 1.0

    return {
        "rpa": pitch_metrics.get("rpa", 0.0),
        "cents_accuracy": np.exp(-cents_error / 500.0),
        "voicing_recall": voicing_metrics.get("recall", 0.0),
        "voicing_precision": voicing_metrics.get("precision", 0.0),
        "octave_accuracy": np.exp(-octave_error_rate * 10.0),
        "gross_error_accuracy": np.exp(-gross_error_rate * 5.0),
        "voicing_f1": voicing_metrics.get("f1", 0.0),
        "rca": pitch_metrics.get("rca", 0.0),
        "rmse_hz": pitch_metrics.get("rmse", np.nan),
        "cents_error": pitch_metrics.get("cents_error", np.nan),
        "octave_error_rate": pitch_metrics.get("octave_error_rate", np.nan),
        "gross_error_rate": pitch_metrics.get("gross_error_rate", np.nan),
    }


def calculate_combined_score(voicing_metrics: Dict, pitch_metrics: Dict) -> float:
    processed = calculate_processed_metrics(voicing_metrics, pitch_metrics)
    components = [
        processed["rpa"],
        processed["cents_accuracy"],
        processed["voicing_recall"],
        processed["voicing_precision"],
        processed["octave_accuracy"],
        processed["gross_error_accuracy"],
    ]
    valid_components = [c for c in components if c and c > 0 and not np.isnan(c)]
    if not valid_components:
        return 0.0
    return len(valid_components) / sum(1.0 / c for c in valid_components)


def run_single_evaluation(
    dataset: Dataset, algorithm_class: object, thresholds: np.ndarray
) -> Dict:
    """
    Runs a full evaluation for a single algorithm on a given dataset.

    This function incorporates:
    1. Strict Failure Handling: If processing any sample fails, the entire
       run for that algorithm is invalidated, and a result with NaN values
       is returned.
    2. Memory Leak Prevention: Explicitly clears GPU cache and calls the
       garbage collector after each sample to prevent memory accumulation.
    """

    def _get_failure_dict() -> Dict:
        """Returns a dictionary structured for a failed run."""
        return {
            "voicing_detection": {
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
            },
            "pitch_accuracy": {
                "rmse": np.nan,
                "cents_error": np.nan,
                "rpa": np.nan,
                "rca": np.nan,
                "octave_error_rate": np.nan,
                "gross_error_rate": np.nan,
                "valid_frames": 0,
            },
            "smoothness_metrics": {
                "relative_smoothness": np.nan,
                "continuity_breaks": np.nan,
            },
            "combined_score": np.nan,
            "optimal_threshold": np.nan,
        }

    def _to_json_safe(d: Dict) -> Dict:
        """Converts numpy types in a dictionary to JSON-serializable types."""
        safe_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                safe_dict[key] = _to_json_safe(value)
            elif isinstance(value, np.integer):
                safe_dict[key] = int(value)
            elif isinstance(value, np.floating):
                safe_dict[key] = None if np.isnan(value) else float(value)
            elif isinstance(value, np.ndarray):
                safe_dict[key] = value.tolist()
            else:
                safe_dict[key] = value
        return safe_dict

    algo_name = algorithm_class.get_name()
    algo = algorithm_class(
        sample_rate=dataset.sample_rate,
        hop_size=dataset.hop_size,
        fmin=dataset.fmin,
        fmax=dataset.fmax,
    )

    threshold_results = {t: defaultdict(list) for t in thresholds}
    skipped_samples = 0
    did_fail = False

    sample_pbar = tqdm(
        range(len(dataset)),
        desc=f"{algo_name}",
        leave=False,
        unit=" samples",
    )

    for idx in sample_pbar:
        try:
            sample = dataset[idx]
            audio = sample["audio"].numpy()
            true_pitch = sample["pitch"].numpy()
            true_voicing = sample["periodicity"].numpy()

            if not true_voicing.any():
                skipped_samples += 1
                continue

            results = algo.extract_pitch(audio, thresholds=list(thresholds))

            if not isinstance(results, list):
                raise TypeError(f"Algorithm returned invalid type: {type(results)}")

            for i, threshold in enumerate(thresholds):
                if i >= len(results):
                    break
                pred_pitch, pred_voicing, _ = results[i]
                data = threshold_results[threshold]
                data["all_pred_voicing"].append(pred_voicing)
                data["all_true_voicing"].append(true_voicing)
                data["all_pred_pitch"].append(pred_pitch)
                data["all_true_pitch"].append(true_pitch)
                data["all_voiced_mask"].append(pred_voicing & true_voicing)
                data["all_smoothness_metrics"].append(
                    evaluate_pitch_smoothness(pred_pitch, pred_voicing, true_voicing)
                )

        except Exception as e:
            tqdm.write(
                f"FATAL: {algo_name} failed on sample {idx}. "
                f"Aborting this algorithm. Error: {e}"
            )
            did_fail = True
            break  # Exit the loop immediately on first failure

        finally:
            # Explicitly clean up memory to prevent leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    sample_pbar.close()

    if did_fail:
        del algo
        gc.collect()
        return _to_json_safe(_get_failure_dict())

    if skipped_samples > 0:
        tqdm.write(f"  ({algo_name} skipped {skipped_samples} unvoiced samples)")

    best_score = -1
    best_metrics = None

    for threshold, data in threshold_results.items():
        if not data["all_pred_voicing"]:
            continue

        global_pred_voicing = np.concatenate(data["all_pred_voicing"])
        global_true_voicing = np.concatenate(data["all_true_voicing"])
        global_pred_pitch = np.concatenate(data["all_pred_pitch"])
        global_true_pitch = np.concatenate(data["all_true_pitch"])
        global_voiced_mask = np.concatenate(data["all_voiced_mask"])

        voicing_metrics = evaluate_voicing_detection(
            global_pred_voicing, global_true_voicing
        )
        pitch_metrics = evaluate_pitch_accuracy(
            global_pred_pitch, global_true_pitch, global_voiced_mask
        )
        smoothness_aggregated = (
            {
                key: np.nanmean([m[key] for m in data["all_smoothness_metrics"] if m])
                for key in data["all_smoothness_metrics"][0]
            }
            if data["all_smoothness_metrics"]
            and data["all_smoothness_metrics"][0] is not None
            else {"relative_smoothness": np.nan, "continuity_breaks": np.nan}
        )
        combined_score = calculate_combined_score(voicing_metrics, pitch_metrics)

        if not np.isnan(combined_score) and combined_score > best_score:
            best_score = combined_score
            best_metrics = {
                "voicing_detection": voicing_metrics,
                "pitch_accuracy": pitch_metrics,
                "smoothness_metrics": smoothness_aggregated,
                "combined_score": combined_score,
                "optimal_threshold": threshold,
            }

    del algo
    gc.collect()

    if best_metrics is None:
        return _to_json_safe(_get_failure_dict())

    return _to_json_safe(best_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single pitch benchmark task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--dataset", type=str, required=True, choices=list_pitch_datasets()
    )
    required.add_argument("--data-dir", type=str, required=True)
    required.add_argument("--chime-dir", type=str, required=True)
    required.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=list_algorithms(),
        choices=list_algorithms(),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=30.0)
    parser.add_argument("--voice-gain-min", type=float, default=-6.0)
    parser.add_argument("--voice-gain-max", type=float, default=6.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-size", type=int, default=256)
    args = parser.parse_args()

    if args.snr_min >= args.snr_max:
        raise ValueError("snr-min must be less than snr-max.")

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    thresholds = np.linspace(0.0, 1.0, 11)

    print(
        f"--- Starting benchmark for dataset '{args.dataset}' with seed {args.seed} ---"
    )
    base_dataset = get_pitch_dataset(args.dataset)(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        hop_size=args.hop_size,
    )
    noisy_dataset = CHiMeNoiseDataset(
        base_dataset=base_dataset,
        chime_home_dir=args.chime_dir,
        background_snr_range=(args.snr_min, args.snr_max),
        voice_gain_range=(args.voice_gain_min, args.voice_gain_max),
        seed=args.seed,   # per-sample deterministic noise = f(seed, idx)
    )

    for algo_name in args.algorithms:
        algo_class = get_algorithm(algo_name)
        # Create a unique string from all key experimental parameters
        param_str = (
            f"sr{int(args.sample_rate / 1000)}k_"  # e.g., sr16k
            f"hop{args.hop_size}_"  # e.g., hop256
            f"snr{int(args.snr_min)}-{int(args.snr_max)}_"  # e.g., snr10-40
            f"gain{int(args.voice_gain_min)}-{int(args.voice_gain_max)}"  # e.g., gain-6-6
        )
        result_path = os.path.join(
            args.output_dir,
            f"{args.dataset}_{algo_name}_{param_str}_seed{args.seed}.json",
        )
        if os.path.exists(result_path):
            tqdm.write(f"Skipping: {os.path.basename(result_path)} already exists.")
            continue

        start_time = time.time()
        metrics = run_single_evaluation(
            dataset=noisy_dataset,
            algorithm_class=algo_class,
            thresholds=thresholds,
        )
        execution_time = time.time() - start_time

        score = metrics.get("combined_score")
        threshold = metrics.get("optimal_threshold")

        tqdm.write(
            f"Finished {algo_name} in {execution_time:.2f}s. "
            f"Score: {score if score is not None else 'N/A':.4f} @ "
            f"Threshold: {threshold if threshold is not None else 'N/A':.2f}"
        )

        full_result = {
            "metadata": {
                "algorithm_name": algo_name,
                "dataset_name": args.dataset,
                "seed": args.seed,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "execution_time_seconds": round(execution_time, 2),
            },
            "parameters": {
                "sample_rate": args.sample_rate,
                "hop_size": args.hop_size,
                "snr_range": (args.snr_min, args.snr_max),
                "voice_gain_range": (args.voice_gain_min, args.voice_gain_max),
            },
            "results": metrics,
        }

        with open(result_path, "w") as f:
            json.dump(full_result, f, indent=4)
        print(f"Success: Saved result to {os.path.basename(result_path)}")

    print(f"\n--- Benchmark run for seed {args.seed} finished. ---")
