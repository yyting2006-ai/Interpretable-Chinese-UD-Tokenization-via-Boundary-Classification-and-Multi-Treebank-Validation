from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


UD_REF = "r2.18"
DATASETS = [
    {
        "name": "UD_Chinese-GSD",
        "repo": "UniversalDependencies/UD_Chinese-GSD",
        "files": {
            "train": "zh_gsd-ud-train.conllu",
            "dev": "zh_gsd-ud-dev.conllu",
            "test": "zh_gsd-ud-test.conllu",
        },
    },
    {
        "name": "UD_Chinese-GSDSimp",
        "repo": "UniversalDependencies/UD_Chinese-GSDSimp",
        "files": {
            "train": "zh_gsdsimp-ud-train.conllu",
            "dev": "zh_gsdsimp-ud-dev.conllu",
            "test": "zh_gsdsimp-ud-test.conllu",
        },
    },
    {
        "name": "UD_Chinese-PUD",
        "repo": "UniversalDependencies/UD_Chinese-PUD",
        "files": {"test": "zh_pud-ud-test.conllu"},
    },
    {
        "name": "UD_Chinese-HK",
        "repo": "UniversalDependencies/UD_Chinese-HK",
        "files": {"test": "zh_hk-ud-test.conllu"},
    },
    {
        "name": "UD_Chinese-CFL",
        "repo": "UniversalDependencies/UD_Chinese-CFL",
        "files": {"test": "zh_cfl-ud-test.conllu"},
    },
]

PUNCTUATION = set("，。！？；：、,.!?;:\"“”‘’（）()《》〈〉【】[]—…·")
CLAUSE_MARKERS = [
    "因为",
    "所以",
    "虽然",
    "但是",
    "可是",
    "如果",
    "那么",
    "同时",
    "然后",
    "而且",
    "不过",
    "于是",
    "一边",
    "一方面",
    "另一方面",
]
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def raw_github_file(repo: str, path: str, data_dir: Path, ref: str = UD_REF) -> str:
    target = data_dir / repo.replace("/", "__") / path
    if target.exists() and target.stat().st_size > 0:
        return target.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            text = subprocess.check_output(
                [
                    "curl",
                    "--http1.1",
                    "-4",
                    "-L",
                    "--fail",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "180",
                    "-s",
                    url,
                ],
                text=True,
            )
            target.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_error = exc
            print(f"download retry {attempt}/3: {url}", flush=True)
            time.sleep(attempt * 2)
    raise RuntimeError(f"failed to download {url}") from last_error


def parse_conllu(text: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    sent_id = ""
    raw_text = ""
    tokens: list[str] = []
    upos_tags: list[str] = []
    deprel_tags: list[str] = []

    def flush() -> None:
        nonlocal sent_id, raw_text, tokens, upos_tags, deprel_tags
        if tokens:
            sentence = "".join(tokens)
            sentences.append(
                {
                    "sent_id": sent_id,
                    "sentence": sentence,
                    "gold_tokens": tokens,
                    "raw_text": raw_text or sentence,
                    "upos_tags": upos_tags,
                    "deprel_tags": deprel_tags,
                }
            )
        sent_id = ""
        raw_text = ""
        tokens = []
        upos_tags = []
        deprel_tags = []

    for line in text.splitlines():
        if not line:
            flush()
            continue
        if line.startswith("# sent_id = "):
            sent_id = line.split("=", 1)[1].strip()
            continue
        if line.startswith("# text = "):
            raw_text = line.split("=", 1)[1].strip()
            continue
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        token_id = parts[0]
        if "-" in token_id or "." in token_id:
            continue
        tokens.append(parts[1])
        upos_tags.append(parts[3])
        deprel_tags.append(parts[7])
    flush()
    return sentences


def load_datasets(data_dir: Path, ref: str = UD_REF) -> dict[str, dict[str, list[dict[str, Any]]]]:
    loaded: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cfg in DATASETS:
        splits = {}
        for split, path in cfg["files"].items():
            text = raw_github_file(cfg["repo"], path, data_dir=data_dir, ref=ref)
            splits[split] = parse_conllu(text)
        loaded[cfg["name"]] = splits
    return loaded


def is_lexical_token(token: str) -> bool:
    if not token or token.isspace():
        return False
    return not all(ch in PUNCTUATION for ch in token)


def tokens_to_boundary_set(tokens: list[str]) -> set[int]:
    boundaries = set()
    pos = 0
    for token in tokens[:-1]:
        pos += len(token)
        boundaries.add(pos)
    return boundaries


def token_spans(tokens: list[str]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for token in tokens:
        end = start + len(token)
        spans.append((start, end))
        start = end
    return spans


def segment_plain_characters(sentence: str) -> list[str]:
    return [ch for ch in sentence if ch.strip()]


def longest_match_segment(sentence: str, lexicon: set[str], protected_units: set[str] | None = None, max_len: int = 6) -> list[str]:
    protected_units = protected_units or set()
    tokens = []
    i = 0
    while i < len(sentence):
        if sentence[i].isspace():
            i += 1
            continue
        best = sentence[i]
        upper = min(max_len, len(sentence) - i)
        for width in range(upper, 0, -1):
            cand = sentence[i : i + width]
            if cand in protected_units or cand in lexicon or cand in CLAUSE_MARKERS:
                best = cand
                break
        tokens.append(best)
        i += len(best)
    return tokens


def char_type(char: str) -> str:
    if not char:
        return "NONE"
    if char in PUNCTUATION:
        return "PUNCT"
    if char.isdigit():
        return "DIGIT"
    if char.isascii() and char.isalpha():
        return "LATIN"
    if CHINESE_RE.search(char):
        return "HAN"
    return "OTHER"


def build_lexicon_from_items(items: list[dict[str, Any]]) -> set[str]:
    counts = Counter(token for item in items for token in item["gold_tokens"] if is_lexical_token(token))
    lexicon = {token for token, _ in counts.items()}
    lexicon.update(CLAUSE_MARKERS)
    return lexicon


def crossing_lexicon_hit(sentence: str, boundary: int, lexicon: set[str], max_len: int = 6) -> bool:
    left = max(0, boundary - max_len)
    right = min(len(sentence), boundary + max_len)
    for start in range(left, boundary):
        for end in range(boundary + 1, right + 1):
            token = sentence[start:end]
            if 2 <= len(token) <= max_len and token in lexicon:
                return True
    return False


def edge_lexicon_hit(sentence: str, boundary: int, lexicon: set[str], side: str, max_len: int = 6) -> bool:
    if side == "left":
        for width in range(1, min(max_len, boundary) + 1):
            if sentence[boundary - width : boundary] in lexicon:
                return True
    else:
        for width in range(1, min(max_len, len(sentence) - boundary) + 1):
            if sentence[boundary : boundary + width] in lexicon:
                return True
    return False


def boundary_features(sentence: str, boundary: int, traditional_boundaries: set[int], lexicon: set[str]) -> dict[str, Any]:
    left_char = sentence[boundary - 1] if boundary > 0 else ""
    right_char = sentence[boundary] if boundary < len(sentence) else ""
    return {
        "left_char": left_char,
        "right_char": right_char,
        "left_type": char_type(left_char),
        "right_type": char_type(right_char),
        "left_right_type": f"{char_type(left_char)}>{char_type(right_char)}",
        "prev2": sentence[max(0, boundary - 2) : boundary],
        "next2": sentence[boundary : min(len(sentence), boundary + 2)],
        "window": sentence[max(0, boundary - 2) : min(len(sentence), boundary + 2)],
        "left_punct": left_char in PUNCTUATION,
        "right_punct": right_char in PUNCTUATION,
        "left_digit": left_char.isdigit(),
        "right_digit": right_char.isdigit(),
        "traditional_boundary": boundary in traditional_boundaries,
        "crossing_lexicon_hit": crossing_lexicon_hit(sentence, boundary, lexicon),
        "left_edge_lexicon_hit": edge_lexicon_hit(sentence, boundary, lexicon, "left"),
        "right_edge_lexicon_hit": edge_lexicon_hit(sentence, boundary, lexicon, "right"),
        "relative_position_bucket": int(10 * boundary / max(1, len(sentence))),
    }


def item_training_rows(item: dict[str, Any], lexicon: set[str]) -> tuple[list[dict[str, Any]], list[int]]:
    sentence = item["sentence"]
    protected_units = {token for token in lexicon if 2 <= len(token) <= 6}
    traditional = longest_match_segment(sentence, lexicon, protected_units=protected_units)
    traditional_boundaries = tokens_to_boundary_set(traditional)
    gold_boundaries = tokens_to_boundary_set(item["gold_tokens"])
    x_rows = []
    y_rows = []
    for boundary in range(1, len(sentence)):
        x_rows.append(boundary_features(sentence, boundary, traditional_boundaries, lexicon))
        y_rows.append(1 if boundary in gold_boundaries else 0)
    return x_rows, y_rows


def train_boundary_classifier(train_items: list[dict[str, Any]], lexicon: set[str]) -> Pipeline:
    x_rows: list[dict[str, Any]] = []
    y_rows: list[int] = []
    for item in train_items:
        x_item, y_item = item_training_rows(item, lexicon)
        x_rows.extend(x_item)
        y_rows.extend(y_item)
    model = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=1.2)),
        ]
    )
    model.fit(x_rows, y_rows)
    return model


def make_tokens(sentence: str, boundaries: set[int]) -> list[str]:
    tokens = []
    start = 0
    for boundary in sorted(boundary for boundary in boundaries if 0 < boundary < len(sentence)):
        if boundary > start:
            tokens.append(sentence[start:boundary])
        start = boundary
    if start < len(sentence):
        tokens.append(sentence[start:])
    return [token for token in tokens if token]


def predict_tokens(sentence: str, model: Pipeline, lexicon: set[str], threshold: float) -> list[str]:
    protected_units = {token for token in lexicon if 2 <= len(token) <= 6}
    traditional = longest_match_segment(sentence, lexicon, protected_units=protected_units)
    traditional_boundaries = tokens_to_boundary_set(traditional)
    if len(sentence) <= 1:
        return [sentence] if sentence else []
    x_rows = [boundary_features(sentence, boundary, traditional_boundaries, lexicon) for boundary in range(1, len(sentence))]
    probs = model.predict_proba(x_rows)[:, 1]
    boundaries = {idx + 1 for idx, prob in enumerate(probs) if prob >= threshold}
    for idx in range(1, len(sentence)):
        if sentence[idx - 1] in PUNCTUATION or sentence[idx] in PUNCTUATION:
            boundaries.add(idx)
    return make_tokens(sentence, boundaries)


def segmentation_metric(pred_tokens: list[str], gold_tokens: list[str]) -> dict[str, float]:
    pred_boundaries = tokens_to_boundary_set(pred_tokens)
    gold_boundaries = tokens_to_boundary_set(gold_tokens)
    pred_spans = set(token_spans(pred_tokens))
    gold_spans = set(token_spans(gold_tokens))

    def prf(pred: set, gold: set) -> tuple[float, float, float]:
        tp = len(pred & gold)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    bp, br, bf = prf(pred_boundaries, gold_boundaries)
    sp, sr, sf = prf(pred_spans, gold_spans)
    return {
        "boundary_precision": bp,
        "boundary_recall": br,
        "boundary_f1": bf,
        "span_precision": sp,
        "span_recall": sr,
        "span_f1": sf,
    }


def relaxed_segmentation_metric(pred_tokens: list[str], gold_tokens: list[str], tolerance: int = 1) -> dict[str, float]:
    pred_boundaries = tokens_to_boundary_set(pred_tokens)
    gold_boundaries = tokens_to_boundary_set(gold_tokens)
    pred_spans = token_spans(pred_tokens)
    gold_spans = token_spans(gold_tokens)

    def relaxed_count(items, refs, matcher) -> int:
        used = set()
        count = 0
        for item in items:
            for idx, ref in enumerate(refs):
                if idx not in used and matcher(item, ref):
                    used.add(idx)
                    count += 1
                    break
        return count

    b_tp = relaxed_count(pred_boundaries, list(gold_boundaries), lambda a, b: abs(a - b) <= tolerance)
    s_tp = relaxed_count(
        pred_spans,
        gold_spans,
        lambda a, b: abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance,
    )

    def prf(tp: int, pred_n: int, gold_n: int) -> tuple[float, float, float]:
        p = tp / pred_n if pred_n else 0.0
        r = tp / gold_n if gold_n else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    bp, br, bf = prf(b_tp, len(pred_boundaries), len(gold_boundaries))
    sp, sr, sf = prf(s_tp, len(pred_spans), len(gold_spans))
    return {
        "boundary_precision": bp,
        "boundary_recall": br,
        "boundary_f1": bf,
        "span_precision": sp,
        "span_recall": sr,
        "span_f1": sf,
    }


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics[0].keys()
    return {key: float(np.mean([m[key] for m in metrics])) for key in keys}


def aggregate_for_items(items: list[dict[str, Any]], mode_to_predictions: dict[str, list[list[str]]]) -> dict[str, dict[str, float]]:
    out = {}
    for mode, predictions in mode_to_predictions.items():
        strict = [segmentation_metric(pred, item["gold_tokens"]) for pred, item in zip(predictions, items)]
        relaxed = [relaxed_segmentation_metric(pred, item["gold_tokens"], tolerance=1) for pred, item in zip(predictions, items)]
        strict_agg = aggregate_metrics(strict)
        relaxed_agg = aggregate_metrics(relaxed)
        out[mode] = {
            "strict_boundary_f1": strict_agg["boundary_f1"],
            "strict_span_f1": strict_agg["span_f1"],
            "relaxed_boundary_f1": relaxed_agg["boundary_f1"],
            "relaxed_span_f1": relaxed_agg["span_f1"],
            "mean_token_count": float(np.mean([len(pred) for pred in predictions])),
        }
    return out


def tune_threshold(dev_items: list[dict[str, Any]], model: Pipeline, lexicon: set[str]) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.25, 0.75, 21):
        predictions = [predict_tokens(item["sentence"], model, lexicon, float(threshold)) for item in dev_items]
        metrics = aggregate_for_items(dev_items, {"ud_boundary_model": predictions})["ud_boundary_model"]
        rows.append({"threshold": round(float(threshold), 3), **metrics})
    df = pd.DataFrame(rows)
    best = df.sort_values(["strict_boundary_f1", "strict_span_f1"], ascending=False).iloc[0]
    return float(best["threshold"]), df


def token_type(token: str) -> str:
    if not token:
        return "EMPTY"
    if all(ch in PUNCTUATION for ch in token):
        return "PUNCT"
    if any(ch.isdigit() for ch in token):
        return "HAS_DIGIT"
    if any(ch.isascii() and ch.isalpha() for ch in token):
        return "HAS_LATIN"
    if all(CHINESE_RE.search(ch) for ch in token):
        return "HAN"
    return "MIXED"


def token_label_features(tokens: list[str], index: int) -> dict[str, Any]:
    token = tokens[index]
    prev_tok = tokens[index - 1] if index > 0 else "<BOS>"
    next_tok = tokens[index + 1] if index + 1 < len(tokens) else "<EOS>"
    return {
        "tok": token,
        "lower": token.lower(),
        "len": min(len(token), 8),
        "type": token_type(token),
        "first": token[:1],
        "last": token[-1:],
        "prefix2": token[:2],
        "suffix2": token[-2:],
        "prev": prev_tok,
        "next": next_tok,
        "prev_type": token_type(prev_tok),
        "next_type": token_type(next_tok),
    }


def train_label_model(items: list[dict[str, Any]], label_key: str) -> Pipeline:
    x_rows = []
    y_rows = []
    for item in items:
        for i, label in enumerate(item[label_key]):
            x_rows.append(token_label_features(item["gold_tokens"], i))
            y_rows.append(label)
    clf = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(max_iter=700, solver="liblinear", class_weight="balanced")),
        ]
    )
    clf.fit(x_rows, y_rows)
    return clf


def predict_token_labels(tokens: list[str], model: Pipeline) -> list[str]:
    if not tokens:
        return []
    return list(model.predict([token_label_features(tokens, i) for i in range(len(tokens))]))


def labeled_span_f1(pred_tokens: list[str], pred_labels: list[str], gold_tokens: list[str], gold_labels: list[str]) -> float:
    pred_labeled = {(s, e, lab) for (s, e), lab in zip(token_spans(pred_tokens), pred_labels)}
    gold_labeled = {(s, e, lab) for (s, e), lab in zip(token_spans(gold_tokens), gold_labels)}
    tp = len(pred_labeled & gold_labeled)
    p = tp / len(pred_labeled) if pred_labeled else 0.0
    r = tp / len(gold_labeled) if gold_labeled else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def get_external_segmenters() -> dict[str, Any]:
    segmenters = {}
    try:
        import jieba

        segmenters["jieba"] = lambda text: [tok for tok in jieba.lcut(text, HMM=True) if tok]
    except Exception as exc:
        print(f"jieba unavailable: {exc}", flush=True)
    try:
        import pkuseg

        seg = pkuseg.pkuseg()
        segmenters["pkuseg"] = lambda text: [tok for tok in seg.cut(text) if tok]
    except Exception as exc:
        print(f"pkuseg unavailable: {exc}", flush=True)
    try:
        import stanza

        nlp = stanza.Pipeline("zh-hans", processors="tokenize", tokenize_no_ssplit=True, verbose=False)

        def stanza_cut(text: str) -> list[str]:
            doc = nlp(text)
            return [tok.text for sent in doc.sentences for tok in sent.tokens if tok.text]

        segmenters["stanza_zh_gsdsimp"] = stanza_cut
    except Exception as exc:
        print(f"stanza unavailable: {exc}", flush=True)
    return segmenters


def train_setting(name: str, train_items: list[dict[str, Any]], dev_items: list[dict[str, Any]]):
    lexicon = build_lexicon_from_items(train_items + dev_items)
    initial = train_boundary_classifier(train_items, lexicon)
    threshold, curve = tune_threshold(dev_items, initial, lexicon)
    final = train_boundary_classifier(train_items + dev_items, lexicon)
    curve["setting"] = name
    return final, lexicon, threshold, curve


def evaluate_predictions(dataset: str, mode: str, items: list[dict[str, Any]], predictions: list[list[str]], upos_model: Pipeline, deprel_model: Pipeline) -> dict[str, Any]:
    strict = aggregate_for_items(items, {mode: predictions})[mode]
    upos_oracle = []
    deprel_oracle = []
    upos_model_scores = []
    deprel_model_scores = []
    for item, pred in zip(items, predictions):
        gold_by_span_upos = {span: label for span, label in zip(token_spans(item["gold_tokens"]), item["upos_tags"])}
        gold_by_span_deprel = {span: label for span, label in zip(token_spans(item["gold_tokens"]), item["deprel_tags"])}
        oracle_upos_labels = [gold_by_span_upos.get(span, "__SEGMENTATION_ERROR__") for span in token_spans(pred)]
        oracle_deprel_labels = [gold_by_span_deprel.get(span, "__SEGMENTATION_ERROR__") for span in token_spans(pred)]
        upos_oracle.append(labeled_span_f1(pred, oracle_upos_labels, item["gold_tokens"], item["upos_tags"]))
        deprel_oracle.append(labeled_span_f1(pred, oracle_deprel_labels, item["gold_tokens"], item["deprel_tags"]))
        upos_model_scores.append(labeled_span_f1(pred, predict_token_labels(pred, upos_model), item["gold_tokens"], item["upos_tags"]))
        deprel_model_scores.append(labeled_span_f1(pred, predict_token_labels(pred, deprel_model), item["gold_tokens"], item["deprel_tags"]))
    return {
        "dataset": dataset,
        "mode": mode,
        "sample_count": len(items),
        **strict,
        "upos_labeled_f1": float(np.mean(upos_oracle)),
        "deprel_labeled_f1": float(np.mean(deprel_oracle)),
        "upos_model_f1": float(np.mean(upos_model_scores)),
        "deprel_model_f1": float(np.mean(deprel_model_scores)),
    }


def run(output_dir: Path, data_dir: Path, ref: str = UD_REF) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_datasets(data_dir=data_dir, ref=ref)
    segmenters = get_external_segmenters()
    settings = {
        "single_gsd": (loaded["UD_Chinese-GSD"]["train"], loaded["UD_Chinese-GSD"]["dev"]),
        "single_gsdsimp": (loaded["UD_Chinese-GSDSimp"]["train"], loaded["UD_Chinese-GSDSimp"]["dev"]),
        "joint_gsd_gsdsimp": (
            loaded["UD_Chinese-GSD"]["train"] + loaded["UD_Chinese-GSDSimp"]["train"],
            loaded["UD_Chinese-GSD"]["dev"] + loaded["UD_Chinese-GSDSimp"]["dev"],
        ),
    }
    trained = {}
    curves = []
    for setting, (train_items, dev_items) in settings.items():
        print(f"training {setting}: train={len(train_items)} dev={len(dev_items)}", flush=True)
        model, lexicon, threshold, curve = train_setting(setting, train_items, dev_items)
        trained[setting] = {"model": model, "lexicon": lexicon, "threshold": threshold}
        curves.append(curve)

    downstream_train = settings["joint_gsd_gsdsimp"][0] + settings["joint_gsd_gsdsimp"][1]
    upos_model = train_label_model(downstream_train, "upos_tags")
    deprel_model = train_label_model(downstream_train, "deprel_tags")

    rows = []
    examples = []
    joint_lexicon = trained["joint_gsd_gsdsimp"]["lexicon"]
    protected = {tok for tok in joint_lexicon if 2 <= len(tok) <= 6}
    for dataset, splits in loaded.items():
        items = splits["test"]
        print(f"evaluating {dataset}: n={len(items)}", flush=True)
        predictions_by_mode: dict[str, list[list[str]]] = {
            "raw_char": [segment_plain_characters(item["sentence"]) for item in items],
            "traditional_joint_lexicon": [
                longest_match_segment(item["sentence"], joint_lexicon, protected_units=protected) for item in items
            ],
        }
        for setting, bundle in trained.items():
            predictions_by_mode[setting] = [
                predict_tokens(item["sentence"], bundle["model"], bundle["lexicon"], bundle["threshold"])
                for item in items
            ]
        for name, func in segmenters.items():
            predictions_by_mode[name] = [func(item["sentence"]) for item in items]
        for mode, predictions in predictions_by_mode.items():
            rows.append(evaluate_predictions(dataset, mode, items, predictions, upos_model, deprel_model))
        for item in items[:25]:
            row = {
                "dataset": dataset,
                "sent_id": item["sent_id"],
                "sentence": item["sentence"],
                "gold": json_dumps(item["gold_tokens"]),
            }
            for setting, bundle in trained.items():
                row[setting] = json_dumps(predict_tokens(item["sentence"], bundle["model"], bundle["lexicon"], bundle["threshold"]))
            examples.append(row)

    summary = pd.DataFrame(rows)
    thresholds = pd.concat(curves, ignore_index=True)
    examples_df = pd.DataFrame(examples)
    summary.to_csv(output_dir / "multitreebank_downstream_summary.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(output_dir / "threshold_curves.csv", index=False, encoding="utf-8-sig")
    examples_df.to_csv(output_dir / "prediction_examples.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "multitreebank_downstream_results.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        thresholds.to_excel(writer, sheet_name="thresholds", index=False)
        examples_df.to_excel(writer, sheet_name="examples", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/ud"))
    parser.add_argument("--ud-ref", default=UD_REF)
    args = parser.parse_args()
    run(output_dir=args.output_dir, data_dir=args.data_dir, ref=args.ud_ref)


if __name__ == "__main__":
    main()

