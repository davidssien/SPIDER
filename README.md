# SPIDER

Two pill-shaped bones connected by one motorized joint. Python trains and serves
three separate PPO policies. The C++ window displays the selected policy live, with the other policies
as semi-transparent bots behind it.

The C++ code stays in `src/main.cpp`, with `Vec2D`, `Bone`, `Joint`, `Skel`, and
`Engine`. Python also launches this executable without a window to run training
physics. This keeps the same mass, inertia, joint impulses, torque, and collisions
in training and playback. There is no second physics implementation in Python.

## Run

You need a C++17 compiler, SDL2, Python, NumPy, and PyTorch. On macOS, install SDL2
with `brew install sdl2` if needed.

```sh
make
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

In another terminal in this folder:

```sh
make run
```

If your Python already has NumPy and PyTorch, you can skip the virtual environment
setup and use `python3 train.py` directly.

- **1 / 2 / 3:** watch a different policy.
- **Space:** freeze the viewer; training and server playback continue.
- **R:** restart the selected policy's demonstration.
- **P:** push its upper bone sideways.
- **Esc:** close the viewer.
- **Ctrl+C in Python:** stop the server and save training progress.

The server listens only on `127.0.0.1:5005`. Start Python and the viewer in either
order. Closing the viewer does not stop training.

## Training

Each policy has a different random seed, a small neural network, and its own
optimizer and checkpoint. Training alternates between the policies; all have
independent demonstrations running at 60 Hz. Updated weights reach the viewer
after each training update.

The observation contains both bones' angles, angular velocities, linear
velocities, lower-bone horizontal position, upper-bone height, and the previous
action. The action sets the joint's target angle between -0.8 and +0.8 radians.
The joint's damped motor converts that target into torque.

Episodes start upright with a small random lean. The reward favors staying alive
and upright, with a small action penalty and a fall penalty. An episode ends
when average uprightness falls below 0.65, the bot moves too far sideways, or
10 seconds elapse. A terminal state resets automatically. The two bones are free
bodies: the bottom is not pinned, and there is no ankle motor.

```sh
python train.py --steps 1000000
python train.py --count 1 --steps 300000
python train.py --no-viewer --steps 1000000
```

`--steps` is the total target per policy, rounded up to a full rollout. Defaults:
three policies, 300,000 steps each, 16 training environments per policy, and
128 steps per rollout. Checkpoints in `models/model1.pt`, `model2.pt`, and
`model3.pt` save after each update. Rerunning resumes weights, optimizer, and
step counts; environments restart. Use `--models another-folder` for a fresh run.
Training stops at its budget and keeps serving the resulting policies until
Ctrl+C. A completed budget does not imply that balancing has been solved.

## Watch saved models or measure progress

```sh
python train.py --serve
python train.py --evaluate --episodes 20
```

Evaluation compares saved policies against zero action on the same held-out
reset sequence, reporting average episode duration and reward. The duration is
capped at 10 seconds. Use the same `--count` and `--models` options as your training
run when loading fewer policies or another checkpoint folder.

The initial saved run trained each policy for 301,056 steps. On 20 held-out
episodes, average balance durations were 7.77 s, 6.87 s, and 5.35 s for models
1–3, versus 0.82 s for zero action (10 s episode limit). These are preliminary
results for this reset distribution, not a guarantee of recovery from pushes.
