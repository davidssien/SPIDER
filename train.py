import argparse
import copy
import json
from pathlib import Path
import socket
import subprocess
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parent
OBS = 13


class Physics:
    def __init__(self, count, seed):
        self.count = count
        self.process = subprocess.Popen(
            [str(ROOT / "app"), "--physics", str(count), str(seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        try:
            self.state = self.command("reset")
        except Exception:
            self.close()
            raise

    def command(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        rows = []
        for _ in range(self.count):
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("C++ physics process stopped")
            row = np.fromstring(line, sep=" ", dtype=np.float32)
            if row.size != 22 or not np.isfinite(row).all():
                raise RuntimeError("Invalid state from C++ physics")
            rows.append(row)
        self.state = np.stack(rows)
        return self.state

    def step(self, actions):
        return self.command("step " + " ".join(str(float(a)) for a in actions))

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 1)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.7]))
        nn.init.normal_(self.actor.weight, std=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, observation):
        features = self.body(observation)
        distribution = Normal(self.actor(features).squeeze(-1), self.log_std.clamp(-3, 0).exp())
        return distribution, self.critic(features).squeeze(-1)

    def action(self, observation):
        distribution, _ = self(observation)
        return distribution.mean.tanh()


class Trainer:
    def __init__(self, index, args):
        self.name = f"model{index + 1}"
        self.seed = args.seed + index
        torch.manual_seed(self.seed)
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.steps = 0
        self.path = args.models / (self.name + ".pt")
        if self.path.exists():
            saved = torch.load(self.path, map_location="cpu", weights_only=True)
            if saved.get("version") != 1:
                raise ValueError(f"Unsupported checkpoint: {self.path}")
            self.policy.load_state_dict(saved["policy"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.steps = saved["steps"]
        elif args.serve or args.evaluate:
            raise FileNotFoundError(f"Train first: missing {self.path}")
        self.physics = None if args.serve or args.evaluate else Physics(args.envs, self.seed)
        self.episode_returns = np.zeros(args.envs)
        self.completed = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        torch.save({"version": 1, "policy": self.policy.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "steps": self.steps,
                    "seed": self.seed}, temporary)
        temporary.replace(self.path)

    def train(self, horizon):
        observations, actions, logs, values, rewards, dones = [], [], [], [], [], []
        for _ in range(horizon):
            observation = torch.from_numpy(self.physics.state[:, :OBS].copy())
            with torch.no_grad():
                distribution, value = self.policy(observation)
                action = distribution.sample()
                log = distribution.log_prob(action)
            state = self.physics.step(action.tanh().numpy())
            observations.append(observation)
            actions.append(action)
            logs.append(log)
            values.append(value)
            rewards.append(torch.from_numpy(state[:, 13].copy()) / 10)
            dones.append(torch.from_numpy(state[:, 14].copy()))
            self.episode_returns += state[:, 13]
            for i in np.flatnonzero(state[:, 14]):
                self.completed.append(float(self.episode_returns[i]))
                self.episode_returns[i] = 0
            self.completed = self.completed[-100:]

        with torch.no_grad():
            _, next_value = self.policy(torch.from_numpy(self.physics.state[:, :OBS].copy()))
        advantages = torch.zeros_like(torch.stack(rewards))
        advantage = torch.zeros(self.physics.count)
        for t in reversed(range(horizon)):
            alive = 1 - dones[t]
            delta = rewards[t] + 0.99 * next_value * alive - values[t]
            advantage = delta + 0.99 * 0.95 * alive * advantage
            advantages[t] = advantage
            next_value = values[t]
        returns = (advantages + torch.stack(values)).flatten()
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        observations = torch.stack(observations).reshape(-1, OBS)
        actions = torch.stack(actions).flatten()
        old_logs = torch.stack(logs).flatten()
        for _ in range(4):
            for batch in torch.randperm(len(actions)).split(256):
                distribution, value = self.policy(observations[batch])
                ratio = (distribution.log_prob(actions[batch]) - old_logs[batch]).exp()
                gain = torch.minimum(ratio * advantages[batch],
                                     ratio.clamp(0.8, 1.2) * advantages[batch])
                loss = -gain.mean() + 0.5 * (value - returns[batch]).square().mean()
                loss -= 0.005 * distribution.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        self.steps += horizon * self.physics.count
        self.save()
        score = np.mean(self.completed) if self.completed else 0
        print(f"{self.name}: {self.steps:,} steps, mean episode reward {score:.1f}", flush=True)


class Server(threading.Thread):
    def __init__(self, trainers):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.models = [copy.deepcopy(t.policy).eval() for t in trainers]
        self.names = [t.name for t in trainers]
        self.steps = [t.steps for t in trainers]
        self.stopping = threading.Event()
        self.error = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("127.0.0.1", 5005))
            self.socket.setblocking(False)
        except Exception:
            self.socket.close()
            raise

    def publish(self, index, trainer):
        with self.lock:
            self.models[index].load_state_dict(trainer.policy.state_dict())
            self.steps[index] = trainer.steps

    def run(self):
        physics = None
        try:
            physics = Physics(len(self.models), 100000)
            print("Viewer server: 127.0.0.1:5005 (run make run)", flush=True)
            while not self.stopping.is_set():
                start = time.monotonic()
                with self.lock, torch.no_grad():
                    actions = [model.action(torch.from_numpy(physics.state[i, :OBS].copy())).item()
                               for i, model in enumerate(self.models)]
                state = physics.step(actions)
                for _ in range(128):
                    try:
                        request, address = self.socket.recvfrom(128)
                    except BlockingIOError:
                        break
                    try:
                        selected, command = request.decode("ascii").split()
                        selected = int(selected)
                        if selected not in range(len(self.models)):
                            continue
                        if command in ("reset", "push"):
                            state = physics.command(f"{command} {selected}")
                        elif command != "watch":
                            continue
                    except (ValueError, UnicodeError):
                        continue
                    for model, row in enumerate(state):
                        fields = ["SPIDER1", model, self.names[model], self.steps[model],
                                  int(row[15]), float(row[13]), *row[16:22]]
                        self.socket.sendto(" ".join(map(str, fields)).encode(), address)
                self.stopping.wait(max(0, 1 / 60 - (time.monotonic() - start)))
        except Exception as error:
            self.error = error
            self.stopping.set()
        finally:
            if physics is not None:
                physics.close()
            self.socket.close()


def evaluate(trainers, episodes):
    results = {}
    for trainer in [None, *trainers]:
        physics = Physics(1, 90000)
        lengths, scores = [], []
        score = 0
        try:
            while len(lengths) < episodes:
                with torch.no_grad():
                    action = 0 if trainer is None else trainer.policy.action(
                        torch.from_numpy(physics.state[0, :OBS].copy())).item()
                row = physics.step([action])[0]
                score += row[13]
                if row[14]:
                    lengths.append(int(row[15]))
                    scores.append(float(score))
                    score = 0
            name = "zero_action" if trainer is None else trainer.name
            results[name] = {"mean_seconds": float(np.mean(lengths) / 60),
                             "mean_reward": float(np.mean(scores)), "episodes": episodes}
        finally:
            physics.close()
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 9 or not 1 <= args.envs <= 256:
        parser.error("count must be 1–9 and envs must be 1–256")
    if min(args.steps, args.horizon, args.episodes) <= 0 or args.envs * args.horizon < 2:
        parser.error("steps, horizon, episodes must be positive; rollout must contain at least 2 samples")
    if args.serve and (args.evaluate or args.no_viewer):
        parser.error("--serve cannot be combined with --evaluate or --no-viewer")
    torch.set_num_threads(1)
    trainers, server = [], None
    try:
        for i in range(args.count):
            trainers.append(Trainer(i, args))
        if args.evaluate:
            evaluate(trainers, args.episodes)
            return
        if not args.no_viewer:
            server = Server(trainers)
            server.start()
        if not args.serve:
            while any(t.steps < args.steps for t in trainers):
                for i, trainer in enumerate(trainers):
                    if trainer.steps >= args.steps:
                        continue
                    trainer.train(args.horizon)
                    if server:
                        if server.error:
                            raise RuntimeError("Viewer server failed") from server.error
                        server.publish(i, trainer)
            print("Training budget complete. Checkpoints saved.", flush=True)
        if server:
            print("Serving models. Ctrl+C to stop.", flush=True)
            while not server.stopping.wait(0.2):
                pass
            if server.error:
                raise RuntimeError("Viewer server failed") from server.error
    except KeyboardInterrupt:
        print("Stopping; saving completed training updates.", flush=True)
    finally:
        if server:
            server.stopping.set()
            server.join(timeout=10)
        for trainer in trainers:
            if trainer.physics:
                trainer.save()
                trainer.physics.close()


if __name__ == "__main__":
    main()
