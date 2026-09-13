# Third-party licences

This repository is licensed under Apache 2.0 (see [LICENSE](LICENSE)). It depends on, or is
designed to load, the following third-party software and weights. **No model weights are
redistributed in this repository**; every checkpoint is downloaded by the user from its upstream
source under the upstream terms.

## Code dependencies

| Component | Used for | Licence | Source |
|---|---|---|---|
| V-JEPA 2 (code) | Encoder architecture and reference for the action-conditioned predictor | MIT | https://github.com/facebookresearch/vjepa2 |
| geoopt | Riemannian optimisers and reference manifold implementations | Apache 2.0 | https://github.com/geoopt/geoopt |
| DINOv2 (code) | Frozen image encoder for the DINO-WM subject | Apache 2.0 | https://github.com/facebookresearch/dinov2 |
| Hydra / OmegaConf | Configuration and sweeps | Apache 2.0 (Hydra), BSD-3 (OmegaConf) | https://github.com/facebookresearch/hydra |
| transformers | Loading V-JEPA 2 and DINOv2 from the Hugging Face Hub | Apache 2.0 | https://github.com/huggingface/transformers |
| PyTorch | Tensors and autograd | BSD-3 | https://github.com/pytorch/pytorch |
| einops, numpy, pandas, matplotlib, huggingface_hub | Utilities | MIT / BSD-3 / BSD-3 / PSF-based / Apache 2.0 | respective repositories |

## Model weights (downloaded by the user, never committed)

| Weights | How this repository uses them | Licence | Redistributed here? |
|---|---|---|---|
| V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`) | Frozen encoder; predictor head is retrained from scratch | Per the model card (code MIT) | No |
| DINOv2 base (`facebook/dinov2-base`) | Frozen encoder for the DINO-WM subject | Apache 2.0 | No |
| NVIDIA Cosmos 3 Nano (`nvidia/Cosmos-3-Nano`) | Inference only: trajectory generation and tokenizer latents for hyperbolicity probing. Never fine-tuned, never modified. | NVIDIA Open Model Licence / OpenMDW 1.1 (gated; accept on the Hub) | No |

If you redistribute a trained predictor head from this repository, it contains none of the
upstream weights (only the head's own parameters) and is covered by this repository's Apache 2.0
licence. Encoder weights are never written to `checkpoints/predictors/`.
