# ExtraExp

This directory contains extra scalability experiments and larger-scale ablation workflows used for CLIP retrieval benchmarking.

## What is inside

- [code](code/) - main scripts for building indexes, running retrieval, and evaluating overlap
- [ablation](ablation/) - ablation study code and related experiment assets
- [Index](Index/) - generated or stored ANN index artifacts
- [output](output/) - experiment outputs and result files

## Purpose

The ExtraExp folder is used for:

- building ANN indexes at scale
- running batch retrieval experiments
- comparing ANN results against flat KNN baselines
- organizing outputs from ablation and high-recall runs

## Notes

- Some paths inside this directory are tied to the local benchmark layout.
- Refer to the README files inside `code/` and `ablation/` for script-level usage.
