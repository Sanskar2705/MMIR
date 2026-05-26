# Scalability Experiments

This directory contains the benchmarking code used to study retrieval performance and scalability for the different embedding families in this project.

## What is inside

- [clip](clip/) - CLIP-based scalability experiments
- [flava](flava/) - FLAVA-based scalability experiments
- [uniir](uniir/) - UniIR-based scalability experiments
- [ExtraExp](ExtraExp/) - additional large-scale and ablation experiments

## Typical contents

Most subfolders include some combination of:

- index building scripts
- retrieval scripts
- quantization or ANN experiment code
- evaluation or test helpers
- experiment-specific output folders

## Notes

- The folders here are organized by model family or experiment type.
- Some subdirectories contain their own README files with more detailed run instructions.
- Generated indexes, outputs, and logs are usually kept beside the scripts that produce them.
