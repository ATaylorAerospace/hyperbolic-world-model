# How Cosmos is used, and how it is not

NVIDIA Cosmos appears in this repository in exactly two roles, both inference-only:
**a data generator** and **a latent-space probe target**. Three rules are non-negotiable:

1. **We never fine-tune, post-train, distil, quantise-for-training, or otherwise modify Cosmos.**
   No optimiser is ever constructed over its parameters. The adapters in
   `src/hyperbolic_world_model/data/cosmos3/` disable gradients before any call and there is no
   training entry point that accepts the model.
2. **We never report Cosmos's generation quality as a result.** No table or figure in the report
   scores the generated videos. Cosmos is an instrument for producing controlled inputs, not a
   subject of the evaluation. If a generated rollout is visibly wrong, the fix is to regenerate or
   exclude it, and the manifest records what was excluded; the quality itself is not a finding.
3. **No Cosmos weights are redistributed.** Users download them from NVIDIA under NVIDIA's terms
   (OpenMDW 1.1 for the Cosmos 3 release, the NVIDIA Open Model License for Cosmos-Predict2.5).
   Generated videos and latents live under `data/` and are gitignored.

## Role 1: action-conditioned trajectory generator

`generate.py` takes a directory of start frames and a JSON file of action sequences and writes
video rollouts plus metadata to `DATA_ROOT/cosmos3_generated/`. Every branch of a prompt shares the
same start frame, so the output contains **branching futures** that real datasets rarely have:
the long-horizon consistency task pairs branches and measures how fast latents diverge.

```bash
bash scripts/generate_cosmos3_trajectories.sh --start-frames prompts/frames --actions prompts/actions.json
# or, to validate the inputs and see the plan without loading a model:
python -m hyperbolic_world_model.data.cosmos3.generate --start-frames ... --actions ... --dry-run
```

Output layout:

```text
DATA_ROOT/cosmos3_generated/
├── rollouts/<prompt_id>/<branch_id>/frames.npz   # frames uint8 (T, H, W, 3), actions float32 (T-1, a)
├── rollouts/<prompt_id>/<branch_id>/rollout.mp4  # preview only, written if torchvision is importable
├── rollouts/<prompt_id>/<branch_id>/meta.json    # labels, seed, model key and sampler settings, timestamp
├── latents/<prompt_id>/<branch_id>.npz           # tokenizer latents float16 (T', D) + meta (after extraction)
├── manifest.jsonl                                # one line per rollout; the dataset reads only this
└── generation_plan.json                          # written by --dry-run
```

The generated videos are then encoded by the frozen V-JEPA 2-AC or DINOv2 encoder like any other
video and used to train and evaluate *our* predictor heads. Cosmos is the data source, not the
subject.

## Role 2: latent-space probe target

`extract_latents.py` runs the Cosmos video tokenizer's **encoder** (the decoder is never loaded)
on every rollout and stores pooled latents. `metrics/gromov_hyperbolicity.py` then estimates their
delta-hyperbolicity. The question is whether a large generative world model's own latent space is
tree-like; the answer is reported next to the same probe on V-JEPA 2-AC and DINOv2 latents, as a
property of the latent spaces, never as a statement about Cosmos's videos.

## Which Cosmos, and how the model is called

The adapters (`CosmosGenerator`, `CosmosTokenizerBackend`) are written against NVIDIA's
`cosmos_predict2` package, the public action-conditioned inference API at the commit we read
(`Video2WorldInference.generate_vid2world(..., action=...)`, chunked exactly as
`cosmos_predict2/action_conditioned.py` does: first frame real, later frames zero, `chunk_size`
actions per call, per-chunk seeds, upstream's stitching rule). The checkpoint is selected by the
`--model` key registered in that package. Today that key is `Cosmos-Predict2.5-2B/robot/action-cond`;
the Cosmos 3 Nano action-conditioned checkpoint is selected the same way once NVIDIA's package
registers it. We could not read a Cosmos 3-specific API from the development environment, so the
adapter's contract is verified against the current public one (`tests/data/test_cosmos3.py`
replays the chunk loop against a fake `generate_vid2world` and checks call shapes, seeds, zero
padding and stitching).

The `cosmos_predict2` package and its GPU stack are **not** dependencies of this repository; install
them per NVIDIA's setup guide on the generation machine. Everything around the model call (input
validation, seeding, manifest, latent pooling, the dataset) runs and is tested without it.

## Reproducibility of generation

Every rollout's `meta.json` and manifest line record the prompt id, branch id, per-branch seed
(derived deterministically from `--seed`, the prompt id and the branch id), the model key, checkpoint
path, sampler settings and timestamp. Re-running with the same inputs regenerates identical
rollouts; `--overwrite` forces it, otherwise already-generated rollouts are skipped.

## Status

Scripts are ready and tested with fakes; **no generation has been run**. Running it requires a GPU
machine with NVIDIA's package installed and the checkpoint accepted on NVIDIA's terms.
