# 2d-grasp

Minimal Gymnasium + Box2D environment for validating force-closure-inspired rewards in dexterous 2D grasping.

## Environment

- ID: `DexGrasp2D-v0`
- Action: `Box(-1, 1, (4,))` joint position-target increments (2 links x 2 fingers)
- Observation: low-dimensional state (finger, object, contact, stage features)
- Task: lift object above threshold and hold stably
- Curriculum: `circle -> box -> polygon`

## Quick start

```powershell
uv sync
uv run python -c "import importlib, gymnasium as gym; importlib.import_module('2d_grasp'); env=gym.make('DexGrasp2D-v0'); obs,_=env.reset(seed=0); print(obs.shape)"
```

## Rendering

Human window:

```powershell
uv run python -c "import importlib, gymnasium as gym; importlib.import_module('2d_grasp'); env=gym.make('DexGrasp2D-v0', render_mode='human'); obs,_=env.reset(seed=0); [env.step(env.action_space.sample()) for _ in range(300)]; env.close()"
```

RGB array frame capture:

```powershell
uv run python -c "import importlib, gymnasium as gym; importlib.import_module('2d_grasp'); env=gym.make('DexGrasp2D-v0', render_mode='rgb_array'); env.reset(seed=0); frame=env.render(); print(frame.shape, frame.dtype); env.close()"
```

If you are running headless, use `render_mode='rgb_array'`.

## Train PPO baseline

```powershell
uv sync --extra train
uv run python -m 2d_grasp.train_sb3_ppo --timesteps 500000 --n-envs 8
```

Optional progress bar (requires `tqdm` and `rich`):

```powershell
uv add tqdm rich
uv run python -m 2d_grasp.train_sb3_ppo --timesteps 500000 --n-envs 8 --progress-bar
```

## Evaluate

```powershell
uv run python -m 2d_grasp.eval --model runs/dexgrasp_ppo/ppo_dexgrasp.zip --stats runs/dexgrasp_ppo/vecnormalize.pkl --episodes 100
```

## Render a trained model

```powershell
uv run python -m 2d_grasp.render_model --model runs/dexgrasp_ppo/ppo_dexgrasp.zip --stats runs/dexgrasp_ppo/vecnormalize.pkl --episodes 3 --render-mode human
```

Headless render smoke test:

```powershell
uv run python -m 2d_grasp.render_model --model runs/dexgrasp_ppo/ppo_dexgrasp.zip --stats runs/dexgrasp_ppo/vecnormalize.pkl --episodes 1 --render-mode rgb_array
```

## Tests

```powershell
uv run pytest -q
```
