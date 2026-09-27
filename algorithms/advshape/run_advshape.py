"""Run asymmetric advantage-shaped self-play on an OpenSpiel game."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np
import pyspiel
import torch

from open_spiel.python.rl_environment import ChanceEventSampler, Environment
import open_spiel.python.rl_agent as rl_agent

from algorithms.advshape.advshape import (
    AdvShapeLearner,
    Transition,
    normalize_advantages,
)
from algorithms.ppo.ppo import PPOAgent, legal_actions_to_mask
from utils import log_to_csv


@dataclass
class Episode:
    traces: tuple[list[Transition], list[Transition]]
    utilities: tuple[float, float]


def make_env(game_name: str, seed: int):
    game = pyspiel.load_game(game_name)
    return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))


class RunAdvShape:
    """Benchmark runner for Century-RL's learner/teacher self-play scheme."""

    def __init__(self, config, game, expl_callback):
        self.config = config.algorithm
        self.meta_config = config
        self.game = game
        self.expl_callback = expl_callback
        self.total_steps_done = 0
        self.update_idx = 0
        self.learner = None
        self.network = None

    def _obs_and_mask(self, time_step, device):
        player = time_step.current_player()
        obs = np.asarray(time_step.observations["info_state"][player], dtype=np.float32)
        obs = torch.as_tensor(obs.reshape(-1), dtype=torch.float32, device=device)
        legal = time_step.observations["legal_actions"][player]
        mask = torch.zeros(self.game.num_distinct_actions(), dtype=torch.bool, device=device)
        mask[legal] = True
        return obs, mask

    def _strategy_for_player(self, env_idx: int, player: int) -> int:
        # strategy 0 is the deployed learner; strategy 1 is the teacher.
        return 0 if player == self.agent_seats[env_idx] else 1

    def _finish_episode(self, env_idx: int, terminal_time_step) -> Episode:
        agent_seat = self.agent_seats[env_idx]
        env_seat = 1 - agent_seat
        utilities = (
            float(terminal_time_step.rewards[agent_seat]),
            float(terminal_time_step.rewards[env_seat]),
        )
        for strategy_id in (0, 1):
            if self.traces[env_idx][strategy_id]:
                self.traces[env_idx][strategy_id][-1].reward = utilities[strategy_id]
        episode = Episode(
            traces=(self.traces[env_idx][0], self.traces[env_idx][1]),
            utilities=utilities,
        )
        self.traces[env_idx] = [[], []]
        # Deterministic seat rotation avoids conflating policy identity with
        # first/second-player advantage.
        self.agent_seats[env_idx] = 1 - self.agent_seats[env_idx]
        return episode

    def _step_envs(self, env_indices: list[int], episodes: list[Episode]):
        if not env_indices:
            return []

        grouped = {0: [], 1: []}
        for env_idx in env_indices:
            ts = self.time_steps[env_idx]
            strategy = self._strategy_for_player(env_idx, ts.current_player())
            grouped[strategy].append(env_idx)

        finished = []
        for strategy_id, indices in grouped.items():
            if not indices:
                continue
            policy = self.learner.policies[strategy_id]
            obs_masks = [self._obs_and_mask(self.time_steps[i], policy.device) for i in indices]
            obs = torch.stack([x[0] for x in obs_masks])
            masks = torch.stack([x[1] for x in obs_masks])
            with torch.no_grad():
                actions, logprobs, _, values, probs = policy.act(obs, masks)

            for j, env_idx in enumerate(indices):
                transition = Transition(
                    obs=obs[j].detach().cpu(),
                    legal_mask=masks[j].detach().cpu(),
                    action=actions[j].detach().cpu(),
                    old_logprob=logprobs[j].detach().cpu(),
                    old_probs=probs[j].detach().cpu(),
                    value=values[j].view(-1)[0].detach().cpu(),
                )
                self.traces[env_idx][strategy_id].append(transition)
                self.time_steps[env_idx] = self.envs[env_idx].step([actions[j].item()])
                self.total_steps_done += 1

                if self.time_steps[env_idx].last():
                    episodes.append(self._finish_episode(env_idx, self.time_steps[env_idx]))
                    finished.append(env_idx)

        return finished

    def _collect_batch(self, requested_steps: int) -> list[Episode]:
        episodes: list[Episode] = []
        target = self.total_steps_done + requested_steps
        active = list(range(len(self.envs)))

        # Fill roughly one PPO batch. Environments that terminate before the
        # target are immediately restarted so all workers stay busy.
        while self.total_steps_done < target:
            finished = self._step_envs(active, episodes)
            for env_idx in finished:
                self.time_steps[env_idx] = self.envs[env_idx].reset()

        # Do not cut trajectories at the batch boundary. Finish every episode
        # that has already started, but do not start a new one. This makes GAE
        # follow each strategy's own decision sequence exactly.
        draining = [
            i for i in active if self.traces[i][0] or self.traces[i][1]
        ]
        while draining:
            finished = set(self._step_envs(draining, episodes))
            next_draining = []
            for env_idx in draining:
                if env_idx in finished:
                    self.time_steps[env_idx] = self.envs[env_idx].reset()
                else:
                    next_draining.append(env_idx)
            draining = next_draining

        return episodes

    def _annotate_trace(self, trace: list[Transition]):
        if not trace:
            return
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        value_lambda = self.config.value_lambda

        next_gae = 0.0
        next_target = 0.0
        for i in reversed(range(len(trace))):
            transition = trace[i]
            value = float(transition.value.item())
            next_value = 0.0 if i == len(trace) - 1 else float(trace[i + 1].value.item())
            reward = float(transition.reward)

            delta = reward + gamma * next_value - value
            transition.advantage = delta + gamma * gae_lambda * next_gae
            transition.value_target = (
                reward
                + gamma * (1.0 - value_lambda) * next_value
                + gamma * value_lambda * next_target
            )
            next_gae = transition.advantage
            next_target = transition.value_target

    @staticmethod
    def _win_rate(utilities):
        if not utilities:
            return 0.5
        outcomes = [1.0 if x > 0 else (0.5 if x == 0 else 0.0) for x in utilities]
        return float(np.mean(outcomes))

    def _prepare(self, episodes: list[Episode]):
        transitions = [[], []]
        utilities = [[], []]
        for episode in episodes:
            for strategy_id in (0, 1):
                trace = episode.traces[strategy_id]
                self._annotate_trace(trace)
                transitions[strategy_id].extend(trace)
                utilities[strategy_id].append(episode.utilities[strategy_id])

        normalize_advantages(transitions[0])
        normalize_advantages(transitions[1])
        win_rates = [self._win_rate(utilities[0]), self._win_rate(utilities[1])]
        return transitions, win_rates

    def _log_update(self, win_rates, metrics, num_episodes):
        log_data = {
            "global_step": self.total_steps_done,
            "episodes": num_episodes,
            "agent_win_rate": win_rates[0],
            "environment_win_rate": win_rates[1],
        }
        for prefix, values in zip(("agent", "environment"), metrics):
            for key, value in values.items():
                log_data[f"{prefix}_{key}"] = value
        log_to_csv(log_data, self.train_log_file)

    def run(self):
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )
        game = self.game
        num_players = game.num_players()
        assert num_players == 2
        assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
        assert game.get_type().reward_model == pyspiel.GameType.RewardModel.TERMINAL

        observation_shape = game.information_state_tensor_shape()
        self.learner = AdvShapeLearner(
            observation_shape=observation_shape,
            num_actions=game.num_distinct_actions(),
            device=device,
            config=self.config,
        )
        self.network = self.learner.agent.network

        self.envs = [
            make_env(str(game), self.meta_config.seed + i)
            for i in range(self.config.num_envs)
        ]
        self.time_steps = [env.reset() for env in self.envs]
        self.agent_seats = [i % 2 for i in range(self.config.num_envs)]
        self.traces = [[[], []] for _ in self.envs]
        self.train_log_file = os.path.join(self.meta_config.experiment_dir, "train_log.csv")

        batch_size = int(self.config.num_envs * self.config.num_steps)
        # Match PPO/MMD's own scheduler denominator for HPO-matched controls.
        num_updates = self.meta_config.max_steps // batch_size + 1
        cp_step = 0
        while self.total_steps_done < self.meta_config.max_steps:
            requested = min(batch_size, self.meta_config.max_steps - self.total_steps_done)
            episodes = self._collect_batch(requested)
            transitions, win_rates = self._prepare(episodes)
            progress = min(self.total_steps_done / self.meta_config.max_steps, 1.0)
            lr_progress = (
                min(self.update_idx / num_updates, 1.0)
                if self.config.lr_schedule == "benchmark_linear"
                else progress
            )

            advantage_scales = (
                self.config.agent_threshold - win_rates[0],
                self.config.environment_threshold - win_rates[1],
            )
            metrics = [
                self.learner.optimize(
                    strategy_id,
                    transitions[strategy_id],
                    advantage_scale=advantage_scales[strategy_id],
                    progress=progress,
                    lr_progress=lr_progress,
                    update_idx=self.update_idx,
                )
                for strategy_id in (0, 1)
            ]
            self.update_idx += 1
            self._log_update(win_rates, metrics, len(episodes))

            while (
                self.total_steps_done >= cp_step + self.meta_config.compute_exploitability_every
            ):
                cp_step += self.meta_config.compute_exploitability_every
                if self.expl_callback is not None:
                    self.expl_callback(
                        self.get_model(), self.get_model(), self.total_steps_done
                    )
                self.save()

        if self.expl_callback is not None:
            self.expl_callback(self.get_model(), self.get_model(), self.total_steps_done)
        self.save()

    def save(self):
        self.learner.agent.save_actor(
            os.path.join(self.meta_config.experiment_dir, "agent.pth")
        )
        self.learner.environment.save_actor(
            os.path.join(self.meta_config.experiment_dir, "environment.pth")
        )

    def current_step(self):
        return self.total_steps_done

    def load_cp(self, cp_path):
        print("loading checkpoint from", cp_path)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )
        self.network = PPOAgent(
            num_actions=self.game.num_distinct_actions(),
            observation_shape=self.game.information_state_tensor_shape(),
            device=device,
        ).to(device)
        self.network.actor.load_state_dict(torch.load(cp_path, map_location=device))

    def wrap_rl_agent(self, *args, **kwargs):
        class AdvShapeRLAgent(rl_agent.AbstractAgent):
            def __init__(self, model, player_id, n_actions):
                self.model = model
                self.player_id = player_id
                self.n_actions = n_actions

            def step(self, time_step, is_evaluation=False):
                obs = time_step.observations["info_state"][self.player_id]
                legal_actions = time_step.observations["legal_actions"][self.player_id]
                legal_mask = torch.zeros((self.n_actions,), dtype=torch.bool)
                legal_mask[legal_actions] = True
                device = next(self.model.parameters()).device
                with torch.no_grad():
                    action = self.model.get_action_and_value(
                        x=torch.tensor(
                            np.asarray(obs), dtype=torch.float32, device=device
                        ),
                        legal_actions_mask=legal_mask.to(device),
                    )[0]
                return rl_agent.StepOutput(action=action.item(), probs=None)

            def get_model(self):
                return self.model.actor

        return [
            AdvShapeRLAgent(
                self.network,
                player_id,
                n_actions=self.game.num_distinct_actions(),
            )
            for player_id in range(self.game.num_players())
        ]

    def get_model(self):
        if self.learner is not None:
            return self.learner.agent.network.actor
        if self.network is not None:
            return self.network.actor
        raise RuntimeError("model is not initialized")
