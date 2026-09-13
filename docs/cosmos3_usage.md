# How Cosmos 3 is used (and how it is not)

NVIDIA Cosmos 3 (the Nano variant) appears in this repository in exactly two roles. Both are
inference-only.

## Role 1: action-conditioned trajectory generator

`src/hyperbolic_world_model/data/cosmos3/generate.py` takes a start frame and an action sequence
and produces a video rollout. We use this to build **controlled branching data**: from one start
frame, several action sequences give several futures that share a prefix. Real datasets rarely
contain such branches; generated ones let the long-horizon consistency task measure how well a
latent space keeps diverging futures apart.

The generated videos are then encoded by the frozen V-JEPA 2 / DINOv2 encoder like any other
video. Cosmos 3 is the *data source*, not the *subject*.

## Role 2: latent-space probe target

`src/hyperbolic_world_model/data/cosmos3/extract_latents.py` runs the Cosmos 3 video tokenizer's
encoder and stores the latents. `metrics/gromov_hyperbolicity.py` then estimates their
delta-hyperbolicity. The question is whether a large generative world model's own latent space is
tree-like. That result is reported alongside the same probe on V-JEPA 2 and DINOv2 latents.

## What is explicitly not done

- Cosmos 3 is **never fine-tuned, distilled, pruned, quantised for training, or otherwise
  modified**. No optimiser is ever constructed over its parameters. The loaders in `data/cosmos3/`
  put the model in `eval()` with gradients disabled and there is no training entry point that
  accepts it.
- No predictor head is attached to Cosmos 3. The "three models" table in the README lists it with
  *Modified: No, frozen* for this reason.
- Its weights are not redistributed. They are downloaded by the user from the Hugging Face Hub
  under NVIDIA's OpenMDW 1.1 terms (`scripts/download_weights.sh cosmos3`, gated, requires
  `HF_TOKEN`). Generated videos and latents are stored under `data/` and are gitignored.
- Cosmos 3 generated videos are not used to train any encoder. They are only encoded by frozen
  encoders and used to train predictor heads.

## Reproducibility of generation

Every generated trajectory records the prompt id, action file, seed and Cosmos 3 revision in
`data/cosmos3/manifest.jsonl`. Generation is seeded per prompt so branching futures can be
regenerated exactly. See `docs/reproducibility.md`.

## Status

Phase 1 (this commit) contains the interfaces and CLI skeletons with `NotImplementedError` and
TODO markers pointing at the model-card API to be checked. Nothing in phase 1 downloads or runs
Cosmos 3.
