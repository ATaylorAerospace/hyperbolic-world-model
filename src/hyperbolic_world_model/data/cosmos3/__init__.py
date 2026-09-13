"""NVIDIA Cosmos 3 as a data source and a latent-space probe target.

Cosmos 3 is **never modified or fine-tuned** here (see ``docs/cosmos3_usage.md``). Two uses:

1. :mod:`generate` - batch script turning (start frame, action sequence) into a video rollout
   with Cosmos 3 Nano, producing controlled trajectories with branching futures.
2. :mod:`extract_latents` - pull Cosmos 3 tokenizer latents so we can measure their
   delta-hyperbolicity independently of any model we train.

:mod:`dataset` loads the generated trajectories in the shared batch format.
"""

from __future__ import annotations

__all__: list[str] = []
