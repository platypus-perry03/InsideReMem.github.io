#!/usr/bin/env python3
"""Frozen cross-dataset behavioral validation for R/M direction components."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DESIGN_PATH = HERE / "design_v2_frozen.json"
AUTH_PATH = HERE / "execution_authorization_v2_frozen.json"
STATIC_REVIEW = HERE / "STATIC_REVIEW_V2.md"
COMPONENT_RE = re.compile(r"^L(\d{2})([HN])(\d{5})$")
POLES = {0: "M", 1: "R"}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("model", "report"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--split", choices=("primary", "confirmation"), default="primary")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--authorization", type=Path, default=AUTH_PATH)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, compression="gzip" if path.suffix == ".gz" else None)
    os.replace(temp, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stable_seed(seed: int, text: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}::{text}".encode()).digest()[:8], "big") % (2**32)


def load_design() -> dict[str, Any]:
    design = read_json(DESIGN_PATH)
    if design.get("study_id") != "cross_dataset_behavioral_validation_v2": raise RuntimeError("Bad study id")
    if design.get("automatic_result_pdf_update_allowed") is not False: raise RuntimeError("PDF mutation must be disabled")
    if design.get("intervention_position") != "last_prompt_token_only": raise RuntimeError("Intervention contract changed")
    if design.get("candidate_alphas") != [0.5, 1.0]: raise RuntimeError("Dose contract changed")
    return design


def model_entry(design: dict[str, Any], name: str) -> dict[str, Any]:
    hits = [row for row in design["models"] if row["name"] == name]
    if len(hits) != 1: raise RuntimeError(f"Unknown model: {name}")
    return hits[0]


def manifest_path(design: dict[str, Any], model: str) -> Path:
    return resolve(design["manifest_root"]) / f"{model}.json"


def validate_manifest(design: dict[str, Any], model: str) -> dict[str, Any]:
    manifest = read_json(manifest_path(design, model))
    if manifest.get("status") != "FROZEN_BEFORE_V2_RESULTS" or manifest.get("model") != model:
        raise RuntimeError("Candidate manifest contract failed")
    seen, means = set(), {}
    for row in manifest["candidates"]:
        cid = row["component_id"]
        if cid in seen or not COMPONENT_RE.fullmatch(cid): raise RuntimeError(f"Bad candidate: {cid}")
        seen.add(cid)
        if not set(row["target_poles"]).issubset({"M", "R"}): raise RuntimeError("Bad target pole")
        for control in (row["matched_control"], row["random_control"]):
            if not COMPONENT_RE.fullmatch(control) or control[3] != cid[3]: raise RuntimeError("Control type mismatch")
        for key, value in row["neuron_means"].items(): means[key] = float(value)
        for key in (cid, row["matched_control"], row["random_control"]):
            if key[3] == "N" and key not in means: raise RuntimeError(f"Missing neuron mean: {key}")
    manifest["neuron_means"] = means
    return manifest


def validate_authorization(path: Path, design: dict[str, Any], model: str) -> None:
    auth = read_json(path)
    if auth.get("status") != "FROZEN_EXECUTION_AUTHORIZED": raise RuntimeError("Not authorized")
    if auth.get("design_sha256") != sha(DESIGN_PATH): raise RuntimeError("Design hash mismatch")
    if auth.get("implementation_sha256") != sha(Path(__file__).resolve()): raise RuntimeError("Implementation hash mismatch")
    if auth.get("static_review_sha256") != sha(STATIC_REVIEW): raise RuntimeError("Static review hash mismatch")
    if auth.get("records_sha256") != sha(resolve(design["dataset_asset"])): raise RuntimeError("Dataset hash mismatch")
    if auth.get("manifest_sha256", {}).get(model) != sha(manifest_path(design, model)): raise RuntimeError("Manifest hash mismatch")
    entry = model_entry(design, model)
    model_path = resolve(entry["model_path"])
    actual_model_lock = {
        "config_sha256": sha(model_path / "config.json"),
        "index_sha256": sha(model_path / "model.safetensors.index.json"),
        "shards": {shard.name: sha(shard) for shard in sorted(model_path.glob("*.safetensors"))},
    }
    if auth.get("model_locks", {}).get(model) != actual_model_lock: raise RuntimeError("Model weight hash mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(entry["gpu"]): raise RuntimeError("Physical GPU mismatch")


def parse_component(component_id: str) -> tuple[str, int, int]:
    hit = COMPONENT_RE.fullmatch(component_id)
    if not hit: raise ValueError(component_id)
    return ("head" if hit.group(2) == "H" else "neuron", int(hit.group(1)), int(hit.group(3)))


def prompt(record: dict[str, Any], design: dict[str, Any]) -> str:
    choices = "\n".join(f"{label}. {value}" for label, value in zip(design["answer_labels"], record["options"]))
    return design["prompt_template"].format(question=record["question"], choices=choices)


def records(design: dict[str, Any], split: str) -> list[dict[str, Any]]:
    selected = []
    for index, row in enumerate(read_json(resolve(design["dataset_asset"]))):
        if row["evaluation_split"] != split: continue
        answer = int(row["answer_index"])
        if len(row["options"]) != 4 or not 0 <= answer < 4: raise RuntimeError("Invalid options")
        selected.append({"row_index": index, "question_id": row["sample_id"], "category": row["dataset"],
                         "dataset": row["dataset"], "group": int(row["pole"] == "R"),
                         "answer_index": answer, "option_count": 4, "prompt": prompt(row, design)})
    expected = int(design[f"{split}_samples_each"]) * 4
    if len(selected) != expected: raise RuntimeError(f"Expected {expected} rows, found {len(selected)}")
    return selected


def label_ids(tokenizer: Any, design: dict[str, Any]) -> list[int]:
    ids = []
    for label in design["answer_labels"]:
        tokens = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(tokens) != 1: raise RuntimeError(f"Non-single-token label {label}: {tokens}")
        ids.append(int(tokens[0]))
    return ids


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size): yield values[start:start + size]


class Intervention:
    def __init__(self, model: Any, component: str, alpha: float, mean: float | None) -> None:
        self.model, self.component, self.alpha, self.mean = model, component, float(alpha), mean
        self.handle: Any | None = None

    def install(self) -> None:
        kind, layer_index, index = parse_component(self.component)
        layer = self.model.model.layers[layer_index]
        module = layer.self_attn.o_proj if kind == "head" else layer.mlp.down_proj
        if kind == "neuron" and self.mean is None: raise RuntimeError("Neuron mean missing")

        def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            value = args[0].clone()
            if kind == "head":
                dim = int(getattr(self.model.config, "head_dim", self.model.config.hidden_size // self.model.config.num_attention_heads))
                value[:, -1, index * dim:(index + 1) * dim] *= 1.0 - self.alpha
            else:
                reference = torch.as_tensor(self.mean, dtype=value.dtype, device=value.device)
                value[:, -1, index] = (1.0 - self.alpha) * value[:, -1, index] + self.alpha * reference
            return (value, *args[1:])

        self.handle = module.register_forward_pre_hook(hook)

    def remove(self) -> None:
        if self.handle is not None: self.handle.remove(); self.handle = None


@torch.inference_mode()
def evaluate(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]], ids: list[int],
             batch_size: int, condition: str, component: str | None = None, alpha: float = 0.0,
             mean: float | None = None) -> pd.DataFrame:
    intervention = Intervention(model, component, alpha, mean) if component else None
    if intervention: intervention.install()
    output_rows, labels = [], torch.tensor(ids, device=device)
    try:
        for batch in batches(rows, batch_size):
            encoded = tokenizer([row["prompt"] for row in batch], padding=True, return_tensors="pt", return_token_type_ids=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :].float()
            option_logits = logits.index_select(1, labels)
            for local, row in enumerate(batch):
                available = option_logits[local]
                logp = F.log_softmax(available, dim=-1)
                answer = row["answer_index"]
                foil = torch.cat((available[:answer], available[answer + 1:])).max()
                output_rows.append({"condition_id": condition, "component_id": component or "baseline", "alpha": alpha,
                    "row_index": row["row_index"], "question_id": row["question_id"], "dataset": row["dataset"],
                    "group": row["group"], "answer_index": answer, "predicted_index": int(available.argmax()),
                    "forced_choice_correct": int(int(available.argmax()) == answer),
                    "correct_probability": float(logp.exp()[answer]), "correct_log_probability": float(logp[answer]),
                    "correct_vs_best_foil_margin": float(available[answer] - foil)})
    finally:
        if intervention: intervention.remove()
    return pd.DataFrame(output_rows)


def bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> list[float]:
    if len(values) == 0: return [float("nan"), float("nan")]
    rng, means = np.random.default_rng(seed), []
    for _ in range(reps): means.append(float(rng.choice(values, len(values), replace=True).mean()))
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def signflip(values: np.ndarray, reps: int, seed: int) -> float:
    observed = float(values.mean())
    if observed <= 0: return 1.0
    rng, exceed = np.random.default_rng(seed), 0
    for _ in range(reps): exceed += float((values * rng.choice([-1., 1.], len(values))).mean()) >= observed
    return (exceed + 1) / (reps + 1)


def bh(values: np.ndarray) -> np.ndarray:
    order, n, out = np.argsort(values), len(values), np.empty(len(values), float)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return out


def mcnemar(base: np.ndarray, changed: np.ndarray) -> float:
    damaged = int(np.sum((base == 1) & (changed == 0))); improved = int(np.sum((base == 0) & (changed == 1)))
    return 1.0 if damaged + improved == 0 else float(stats.binomtest(damaged, damaged + improved, .5, alternative="greater").pvalue)


def condition_plan(manifest: dict[str, Any], design: dict[str, Any]) -> list[dict[str, Any]]:
    plan, seen, means = [], set(), manifest["neuron_means"]
    for row in manifest["candidates"]:
        for alpha in design["candidate_alphas"]:
            key = (row["component_id"], float(alpha))
            if key not in seen: plan.append({"condition": f"candidate::{key[0]}::a{alpha:g}", "component": key[0], "alpha": alpha, "mean": means.get(key[0])}); seen.add(key)
        for kind in ("matched_control", "random_control"):
            key = (row[kind], 1.0)
            if key not in seen: plan.append({"condition": f"control::{key[0]}::a1", "component": key[0], "alpha": 1.0, "mean": means.get(key[0])}); seen.add(key)
    return plan


def baseline_summary(frame: pd.DataFrame, design: dict[str, Any]) -> dict[str, Any]:
    groups, passed = {}, True
    margin = float(design["baseline_gate"]["minimum_accuracy_above_uniform_chance_each_dataset"])
    for dataset, subset in frame.groupby("dataset"):
        accuracy = float(subset["forced_choice_correct"].mean())
        row = {"n": len(subset), "correct": int(subset["forced_choice_correct"].sum()), "accuracy": accuracy,
               "uniform_chance": .25, "mean_correct_probability": float(subset["correct_probability"].mean()),
               "pass": accuracy >= .25 + margin}
        groups[dataset] = row; passed &= row["pass"]
    return {"status": "PASS" if passed else "FAIL", "datasets": groups}


def aligned(frame: pd.DataFrame, condition: str, indices: list[int]) -> pd.DataFrame:
    return frame[frame["condition_id"] == condition].set_index("row_index").loc[indices]


def analyze(frame: pd.DataFrame, manifest: dict[str, Any], design: dict[str, Any]) -> pd.DataFrame:
    baseline = frame[frame["condition_id"] == "baseline"].set_index("row_index")
    rows, reps, permutations, seed = [], int(design["bootstrap_replicates"]), int(design["permutation_replicates"]), int(design["dataset_seed"])
    for candidate in manifest["candidates"]:
        cid = candidate["component_id"]
        for group, pole in POLES.items():
            indices = baseline[baseline["group"] == group].index.tolist(); base = baseline.loc[indices]
            half = aligned(frame, f"candidate::{cid}::a0.5", indices); full = aligned(frame, f"candidate::{cid}::a1", indices)
            matched = aligned(frame, f"control::{candidate['matched_control']}::a1", indices)
            random = aligned(frame, f"control::{candidate['random_control']}::a1", indices)
            acc = base["forced_choice_correct"].to_numpy(float) - full["forced_choice_correct"].to_numpy(float)
            half_prob = base["correct_probability"].to_numpy(float) - half["correct_probability"].to_numpy(float)
            prob = base["correct_probability"].to_numpy(float) - full["correct_probability"].to_numpy(float)
            mprob = base["correct_probability"].to_numpy(float) - matched["correct_probability"].to_numpy(float)
            rprob = base["correct_probability"].to_numpy(float) - random["correct_probability"].to_numpy(float)
            key = f"{cid}::{pole}"
            rows.append({"component_id": cid, "pole": pole, "is_target_pole": pole in candidate["target_poles"], "n": len(indices),
                "baseline_correct": int(base["forced_choice_correct"].sum()), "alpha_1_correct": int(full["forced_choice_correct"].sum()),
                "accuracy_drop": float(acc.mean()), "accuracy_drop_ci": bootstrap_ci(acc, reps, stable_seed(seed, key+"acc")),
                "accuracy_p": mcnemar(base["forced_choice_correct"].to_numpy(int), full["forced_choice_correct"].to_numpy(int)),
                "probability_drop_alpha_0_5": float(half_prob.mean()), "probability_drop": float(prob.mean()),
                "probability_drop_ci": bootstrap_ci(prob, reps, stable_seed(seed, key+"prob")),
                "probability_p": signflip(prob, permutations, stable_seed(seed, key+"sign")),
                "candidate_minus_matched_ci": bootstrap_ci(prob-mprob, reps, stable_seed(seed, key+"matched")),
                "candidate_minus_random_ci": bootstrap_ci(prob-rprob, reps, stable_seed(seed, key+"random")),
                "probability_dose_monotonic": bool(0 <= half_prob.mean() <= prob.mean())})
    result = pd.DataFrame(rows)
    result["accuracy_q"] = bh(result["accuracy_p"].to_numpy()); result["probability_q"] = bh(result["probability_p"].to_numpy())
    alpha = float(design["fdr_alpha"])
    result["accuracy_signal"] = result["accuracy_drop_ci"].map(lambda x: x[0] > 0) & (result["accuracy_q"] < alpha)
    result["probability_signal"] = (result["probability_drop_ci"].map(lambda x: x[0] > 0) & (result["probability_q"] < alpha)
        & result["probability_dose_monotonic"] & result["candidate_minus_matched_ci"].map(lambda x: x[0] > 0)
        & result["candidate_minus_random_ci"].map(lambda x: x[0] > 0))
    drops = {(row.component_id, row.pole): row.accuracy_drop for row in result.itertuples()}
    result["target_specificity"] = result.apply(lambda row: (not row["is_target_pole"]) or row["accuracy_drop"] >= drops[(row["component_id"], "M" if row["pole"] == "R" else "R")], axis=1)
    result["strict_target_signal"] = result["is_target_pole"] & result["accuracy_signal"] & result["probability_signal"] & result["target_specificity"]
    result["probability_only_target_signal"] = result["is_target_pole"] & ~result["accuracy_signal"] & result["probability_signal"]
    return result


def load_model(entry: dict[str, Any], device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    path = resolve(entry["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id; tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval().to(device)
    c = model.config
    if c.model_type not in {"llama", "mistral", "olmo2", "gemma2"}: raise RuntimeError("Unsupported architecture")
    return model, tokenizer, {"model_type": c.model_type, "layers": int(c.num_hidden_layers), "heads": int(c.num_attention_heads)}


def execute(ns: argparse.Namespace, design: dict[str, Any]) -> None:
    entry = model_entry(design, ns.model); validate_authorization(ns.authorization, design, ns.model)
    output = resolve(design["output_root"]) / design["run_id"] / ns.model / ns.split
    if output.exists(): raise RuntimeError(f"Refusing overwrite: {output}")
    output.mkdir(parents=True); write_json(output / "status.json", {"status": "RUNNING_BASELINE"})
    rows, manifest, device = records(design, ns.split), validate_manifest(design, ns.model), torch.device(ns.device)
    model = None
    try:
        model, tokenizer, contract = load_model(entry, device); ids = label_ids(tokenizer, design)
        rows.sort(key=lambda row: len(tokenizer(row["prompt"])["input_ids"]))
        print(f"[baseline] {ns.model} {ns.split}: {len(rows)} rows", flush=True)
        base = evaluate(model, tokenizer, device, rows, ids, int(entry["batch_size"]), "baseline")
        gate = baseline_summary(base, design); atomic_csv(output / "baseline_items.csv.gz", base); write_json(output / "baseline_summary.json", gate)
        if gate["status"] != "PASS":
            summary = {"status": "BASELINE_FAIL", "model": ns.model, "split": ns.split, "baseline": gate,
                       "strict_components": [], "probability_only_components": []}
            write_json(output / "summary.json", summary); write_json(output / "status.json", {"status": "BASELINE_FAIL"}); return
        frames, plan = [base], condition_plan(manifest, design); write_json(output / "status.json", {"status": "RUNNING_INTERVENTIONS", "conditions": len(plan)})
        for number, condition in enumerate(plan, 1):
            print(f"[{number}/{len(plan)}] {condition['condition']}", flush=True)
            frames.append(evaluate(model, tokenizer, device, rows, ids, int(entry["batch_size"]), condition["condition"], condition["component"], condition["alpha"], condition["mean"]))
        responses = pd.concat(frames, ignore_index=True); atomic_csv(output / "item_responses.csv.gz", responses)
        result = analyze(responses, manifest, design); serial = result.copy()
        for column in [c for c in serial if c.endswith("_ci")]: serial[column] = serial[column].map(json.dumps)
        atomic_csv(output / "candidate_results.csv", serial)
        strict = sorted(result[result["strict_target_signal"]]["component_id"].unique())
        probability = sorted(result[result["probability_only_target_signal"]]["component_id"].unique())
        summary = {"status": "COMPLETE", "model": ns.model, "split": ns.split, "baseline": gate,
                   "candidate_count": len(manifest["candidates"]), "strict_components": strict,
                   "probability_only_components": probability, "contract": contract,
                   "claim_boundary": design["claim_boundary"], "environment": {"python": platform.python_version(), "torch": torch.__version__}}
        write_json(output / "summary.json", summary); write_json(output / "status.json", {"status": "COMPLETE", "summary_sha256": sha(output / "summary.json")})
        print(f"[complete] strict={strict} probability_only={probability}", flush=True)
    except Exception as error:
        write_json(output / "status.json", {"status": "ERROR", "type": type(error).__name__, "error": str(error)}); raise
    finally:
        if model is not None: model.to("cpu"); del model
        gc.collect(); torch.cuda.empty_cache()


def report(design: dict[str, Any]) -> None:
    root = resolve(design["output_root"]) / design["run_id"]
    summaries = []
    for path in sorted(root.glob("*/*/summary.json")): summaries.append(read_json(path))
    lines = ["# Cross-dataset Behavioral Validation v2 결과", "", f"실행 상태 요약: {len(summaries)}개 model/split 결과", ""]
    for row in summaries:
        lines.append(f"- {row['model']} / {row['split']}: **{row['status']}**, strict={row.get('strict_components', [])}, probability-only={row.get('probability_only_components', [])}")
    lines += ["", "## 해석 제한", "", "- C-Eval-H=M, 세 수학 데이터=R은 LiReF의 task-family 정의이며 문항별 memory_reason_score가 아니다.",
              "- 객관식화한 frozen answer-decision format에서의 결과다.", "- 이 보고서는 result.pdf를 자동 수정하지 않는다."]
    root.mkdir(parents=True, exist_ok=True); (root / "RESULTS_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ns, design = args(), load_design()
    if ns.phase == "model":
        if not ns.model: raise RuntimeError("--model required")
        execute(ns, design)
    else: report(design)


if __name__ == "__main__": main()
