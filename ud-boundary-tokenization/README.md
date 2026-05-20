# Interpretable Chinese UD Tokenization via Boundary Classification

This repository contains code and reproducibility materials for the paper:

> **Interpretable Chinese UD Tokenization via Boundary Classification and Multi-Treebank Validation**

The project implements a lightweight **UD Boundary Model** for Chinese tokenization under the Universal Dependencies (UD) framework. The model treats tokenization as character-boundary classification and evaluates it across multiple Chinese UD treebanks, downstream syntactic labels, and mainstream Chinese segmenters.

## What Is Included

- A self-contained implementation of the UD Boundary Model.
- Scripts to download official UD Chinese treebanks and rerun the experiments.
- Result CSV files used in the paper.
- Publication-style figures for the Results section.
- Documentation of external segmenter settings and UD data sources.

Raw UD treebank files are **not committed** to this repository. The scripts download them from the official Universal Dependencies GitHub repositories using the `r2.18` release tag.

## Data

The experiments use Universal Dependencies release **2.18**:

- Release: Universal Dependencies 2.18
- Release date: 15 May 2026
- Repository archive: LINDAT/CLARIAH-CZ, <http://hdl.handle.net/11234/1-6149>
- License: CC BY-SA 4.0
- Treebanks:
  - `UD_Chinese-GSD`
  - `UD_Chinese-GSDSimp`
  - `UD_Chinese-PUD`
  - `UD_Chinese-HK`
  - `UD_Chinese-CFL`

The default scripts fetch files from the corresponding `UniversalDependencies/*` GitHub repositories at tag `r2.18`.

## Repository Layout

```text
.
├── src/ud_boundary_tokenization/
│   ├── __init__.py
│   └── experiment.py
├── scripts/
│   ├── run_experiments.py
│   └── make_chapter5_figures.py
├── results/
│   ├── multitreebank_downstream_summary.csv
│   ├── paper_key_deltas.csv
│   ├── downstream_bootstrap_ci.csv
│   ├── ablation_summary.csv
│   └── boundary_error_summary.csv
├── figures/
├── docs/
├── data/
│   └── README.md
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Optional external segmenters:

- `jieba`
- `pkuseg`
- `stanza`

If an external segmenter is unavailable, the experiment script skips it and prints a warning.

## Reproducing the Main Experiments

```bash
python scripts/run_experiments.py
```

This command downloads the UD data into `data/ud/`, trains the three configurations of the UD Boundary Model, evaluates internal baselines and available external segmenters, and writes results to `results/`.

The three training configurations are:

| Configuration | Meaning |
|---|---|
| `single_gsd` | UD Boundary Model trained on GSD |
| `single_gsdsimp` | UD Boundary Model trained on GSDSimp |
| `joint_gsd_gsdsimp` | UD Boundary Model trained on GSD + GSDSimp |

These names are training configurations, not different model architectures.

## External Segmenter Settings

The paper uses the following settings:

| System | Version used in paper | Mode / model | Parameters |
|---|---:|---|---|
| jieba | 0.42.1 | default dictionary, accurate mode | `jieba.lcut(text, HMM=True)` |
| pkuseg | 0.0.25 | default segmentation model | `pkuseg.pkuseg(model_name="default", user_dict="default", postag=False)` |
| Stanza | 1.11.1 | `zh-hans` tokenizer, `gsdsimp` package | `stanza.Pipeline("zh-hans", processors="tokenize", tokenize_no_ssplit=True)` |

No simplified/traditional conversion is applied before evaluation. All systems receive the same raw sentence strings reconstructed from UD gold tokens.

## Regenerating Figures

```bash
python scripts/make_chapter5_figures.py
```

The script reads the CSV files in `results/` and writes publication-style figures into `figures/`.

## Key Results

UD Boundary Model-Joint achieves strong strict Boundary-F1 across Chinese UD treebanks:

| Treebank | Strict Boundary-F1 | Strict Span-F1 |
|---|---:|---:|
| GSD | 0.9346 | 0.8409 |
| GSDSimp | 0.9344 | 0.8404 |
| PUD | 0.9316 | 0.8322 |
| HK | 0.9198 | 0.8032 |
| CFL | 0.9186 | 0.7982 |

Sentence-level paired bootstrap resampling shows that downstream gains over the traditional longest-match baseline are statistically reliable across all treebanks and downstream metrics (`p < 0.0001`).

## License

The code in this repository is released under the MIT License.

The Universal Dependencies datasets are not redistributed here. Please refer to the official UD release and individual treebank licenses.

