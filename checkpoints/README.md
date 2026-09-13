# checkpoints/

Gitignored except for this file. Override the root with `CKPT_ROOT` in `.env`.

```text
checkpoints/
├── encoders/                   # frozen upstream weights, downloaded by scripts/download_weights.sh
│   ├── vjepa2_ac/              # vjepa2-ac-vitg.pt fetched by load_vjepa2_ac (torch.hub code cache lives in TORCH_HOME)
│   ├── dinov2/                 # facebook/dinov2-base
│   └── cosmos3-nano/           # nvidia Cosmos 3 Nano, generation and tokenizer only, never trained
└── predictors/                 # trained predictor heads, one directory per Hydra run
    └── <experiment>/<run_id>/
        ├── head.pt             # predictor head state dict (encoder weights are never saved here)
        ├── config.yaml         # resolved Hydra config
        └── metrics.json        # final metrics in the model's native geometry
```

The frozen-encoder invariant is enforced at training time: `train_predictor.py` asserts that no
encoder parameter has `requires_grad=True` before the first optimiser step, and only the predictor
head state dict is ever written to `predictors/`.
