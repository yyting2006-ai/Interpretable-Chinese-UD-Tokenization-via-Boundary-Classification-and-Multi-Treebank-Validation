# Reproducibility Notes

## UD Data

The experiments are designed for Universal Dependencies release `r2.18`.

The scripts download `.conllu` files from:

- `UniversalDependencies/UD_Chinese-GSD`
- `UniversalDependencies/UD_Chinese-GSDSimp`
- `UniversalDependencies/UD_Chinese-PUD`
- `UniversalDependencies/UD_Chinese-HK`
- `UniversalDependencies/UD_Chinese-CFL`

The raw treebanks are not redistributed in this repository.

## Input Normalization

No simplified/traditional conversion is applied. Every system receives the same raw sentence reconstructed by concatenating the UD gold tokens. Predictions are evaluated against the original character offsets.

## External Segmenters

The paper used:

- jieba 0.42.1 with `jieba.lcut(text, HMM=True)`
- pkuseg 0.0.25 with `pkuseg.pkuseg()`
- Stanza 1.11.1 with `stanza.Pipeline("zh-hans", processors="tokenize", tokenize_no_ssplit=True)`

Stanza loads the `zh-hans` tokenizer with the default `gsdsimp` package.

## Downstream Evaluation

UPOS and dependency relation labels are evaluated using labelled span-F1. A predicted item is correct only if both the character span and the label match the gold annotation.

The downstream classifiers are logistic regression models trained on gold-tokenized GSD + GSDSimp train/dev data.

## Bootstrap Confidence Intervals

The reported downstream confidence intervals were computed with sentence-level paired bootstrap resampling:

- comparison: UD Boundary Model-Joint vs traditional longest-match
- resampling unit: sentence
- number of bootstrap samples: 10,000
- random seed: 20260520
- interval: percentile 95% confidence interval

