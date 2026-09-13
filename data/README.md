# data/

This directory is gitignored except for this file. Nothing here is committed; everything is
regenerable from the scripts in `scripts/`.

Expected layout (override the root with `DATA_ROOT` in `.env`):

```text
data/
├── droid/                      # DROID (Khazatsky et al., 2024) episodes, RLDS or LeRobot format
│   ├── metadata.json           # per-episode embodiment / task / primitive labels
│   └── episodes/               # one shard per episode: frames + 7-DoF actions
├── cosmos3/
│   ├── prompts/                # start frames + action sequences used as generation inputs
│   ├── rollouts/               # generated .mp4 videos from Cosmos 3 Nano (gitignored by extension)
│   ├── latents/                # Cosmos 3 tokenizer latents (.npz) used for hyperbolicity probing
│   └── manifest.jsonl          # one line per trajectory: prompt id, action file, video path, seed
└── hierarchies/
    └── droid_tree.json         # embodiment > task > primitive tree built by data/hierarchies.py
```

Populate with:

```bash
bash scripts/download_weights.sh                # weights only, no data
bash scripts/generate_cosmos3_trajectories.sh   # writes data/cosmos3/
```

DROID itself must be downloaded separately following the instructions in
`src/hyperbolic_world_model/data/droid.py`.
