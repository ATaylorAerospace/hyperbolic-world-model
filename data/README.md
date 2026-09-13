# data/

This directory is gitignored except for this file. Nothing here is committed; everything is
regenerable from the scripts in `scripts/`.

Expected layout (override the root with `DATA_ROOT` in `.env`):

```text
data/
├── droid/                      # DROID (Khazatsky et al., 2024) episodes, RLDS or LeRobot format
│   ├── metadata.json           # per-episode embodiment / task / primitive labels
│   └── episodes/               # one shard per episode: frames + 7-DoF actions
├── cosmos3_prompts/            # your generation inputs (any location; passed to generate.py)
│   ├── frames/                 # start frames: <prompt_id>.png|.jpg|.npy (H, W, 3) uint8
│   └── actions.json            # {"action_dim": 10, "prompts": [{prompt_id, start_frame, embodiment, task, prompt,
│                               #   branches: [{branch_id, primitive, actions: [[10 floats: droid_lerobot], ...]}]}]}
├── cosmos3_generated/          # written by generate.py / extract_latents.py
│   ├── rollouts/<prompt_id>/<branch_id>/{frames.npz, meta.json, rollout.mp4?}
│   ├── latents/<prompt_id>/<branch_id>.npz
│   ├── manifest.jsonl          # one line per rollout; the dataset reads only this
│   └── generation_plan.json    # from --dry-run
└── hierarchies/
    └── droid_tree.json         # embodiment > task > primitive tree built by data/hierarchies.py
```

Populate with:

```bash
bash scripts/download_weights.sh                # weights only, no data
bash scripts/generate_cosmos3_trajectories.sh --start-frames data/cosmos3_prompts/frames --actions data/cosmos3_prompts/actions.json   # writes data/cosmos3_generated/
```

DROID itself must be downloaded separately following the instructions in
`src/hyperbolic_world_model/data/droid.py`.
