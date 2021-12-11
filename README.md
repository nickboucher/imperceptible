# Imperceptible Perturbations
[*Bad Characters: Imperceptible NLP Attacks*](https://arxiv.org/abs/2106.09898)

## Overview

Several years of research have shown that machine-learning systems are vulnerable to adversarial examples, both in theory and in practice. Until now, such attacks have primarily targeted visual models, exploiting the gap between human and machine perception. Although text-based models have also been attacked with adversarial examples, such attacks struggled to preserve semantic meaning and indistinguishability.

In our [related paper](https://arxiv.org/abs/2106.09898), we explore a large class of adversarial examples that can be used to attack text-based models in a black-box setting without making any human-perceptible visual modification to inputs. We use encoding-specific perturbations that are imperceptible to the human eye to manipulate the outputs of a wide range of Natural Language Processing (NLP) systems from neural machine-translation pipelines to web search engines. We find that with a single imperceptible encoding injection -- representing one invisible character, homoglyph, reordering, or deletion -- an attacker can significantly reduce the performance of vulnerable models, and with three injections most models can be functionally broken.

Our attacks work against currently-deployed commercial systems, including those produced by Microsoft and Google, in addition to open source models published by Facebook, IBM, and HuggingFace. This novel series of attacks presents a significant threat to many language processing systems: an attacker can affect systems in a targeted manner without any assumptions about the underlying model.

We conclude that text-based NLP systems require careful input sanitization, just like conventional applications, and that given such systems are now being deployed rapidly at scale, the urgent attention of architects and operators is required.

## This Respository

This repository contains all accompanying code and results referenced in [Bad Characters: Imperceptible NLP Attacks](https://arxiv.org/abs/2106.09898).

A website summarizing the findings of the paper can be found at [imperceptible.ml](https://imperceptible.ml).

### Repository Layout

A Python command line utility to reproduce the experiments referenced in the paper can be found in the `experiments/` directory.

IPython notebooks providing walkthroughs of the results in the paper can be found in the `notebooks/` directory.

The results of our experiments can be found serialized as Python pickles and JSON files in the `results/` directory along an IPython notebook that can be used to reproduce all of the figures in the paper.

A website to convey the results of the paper and provide adversarial example generation/validation tools can be found in the `website/` directory. A live version can be found at [imperceptible.ml](https://imperceptible.ml).

## Citation

If you use anything in this repository, in the [*Bad Characters*](https://arxiv.org/abs/2106.09898) paper, or on [imperceptible.ml](https://imperceptible.ml) in your own work, please cite the following:

```bibtex
@article{boucher_imperceptible_2021,
    title = {Bad {Characters}: {Imperceptible} {NLP} {Attacks}},
    url = {https://arxiv.org/abs/2106.09898},
    journal = {Preprint.},
    author = {Nicholas Boucher and Ilia Shumailov and Ross Anderson and Nicolas Papernot},
    year = {2021}
}
```
