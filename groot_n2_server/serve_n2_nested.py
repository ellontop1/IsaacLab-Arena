#!/usr/bin/env python
"""Serve the N2 policy for IsaacLab-Arena (nested GR00T protocol).

Arena's ``Gr00tRemoteClosedloopPolicy`` sends observations in the *nested*
GR00T format ``{"video": {cam: ...}, "state": {grp: ...}, "language": {...}}``
and expects the action dict keyed by joint-group names (``left_arm`` ...),
shaped ``(N, horizon, D)``.

The stock ``python -m groot.infra.oss policy-server`` unconditionally wraps the
policy in ``Gr00tFlatPolicyWrapper`` (see run_gr00t_server.main line ~284),
which instead expects *flat* dotted keys (``video.ego_view``) and returns
``action.<key>`` with the batch dim squeezed out. That is the exact
client/server mismatch we hit with Arena.

This launcher reuses ``run_gr00t_server.main`` but monkeypatches the flat
wrapper with a passthrough adapter that:
  * forwards Arena's nested observation straight to the raw Gr00tN2Policy
    (which already expects nested input), and
  * augments the returned action with zero ``navigate_command`` (3) and
    ``base_height_command`` (1), which Arena's G1_LOCOMANIPULATION action
    builder requires but our static-trained model does not emit. Zeros == no
    locomotion, which is correct for the static pick-and-place task.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

import groot.core.policy.run_gr00t_server as rgs
from groot.core.data import schema
from groot.core.policy import base


class NestedNavAdapter(base.PolicyWrapper):
    """Passthrough for nested obs + zero WBC (navigate/base_height) commands."""

    def check_observation(self, observation: dict[str, Any]) -> None:  # noqa: D102
        pass

    def check_action(self, action: dict[str, Any]) -> None:  # noqa: D102
        pass

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.policy.get_action(observation, options)

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action, info = self.policy.get_action(observation, options)
        if not action:
            return action, info
        self._log_action_values(observation, action)
        # Every joint group is (N, horizon, D); reuse its leading dims for the
        # zero WBC commands so Arena can concatenate along the last axis.
        sample = next(iter(action.values()))
        lead = np.asarray(sample).shape[:-1]
        if "navigate_command" not in action:
            action["navigate_command"] = np.zeros((*lead, 3), dtype=np.float32)
        if "base_height_command" not in action:
            action["base_height_command"] = np.zeros((*lead, 1), dtype=np.float32)
        return action, info

    @staticmethod
    def _log_action_values(observation: dict[str, Any], action: dict[str, Any]) -> None:
        """Log per-group action stats to tell 'reaching' from 'hold-like'.

        For each joint group:
          span = max over joints of (max_t - min_t) across the predicted horizon
                 (~0 => flat chunk, i.e. no motion planned within the chunk)
          d0   = max |first_predicted_target - current_state| for that group
                 (~0 => commanding the robot to stay where it already is)
          amax = max |target| (sanity: non-degenerate magnitudes)
        """
        state = observation.get("state", {}) if isinstance(observation, dict) else {}
        parts: list[str] = []
        for key, val in action.items():
            a = np.asarray(val, dtype=np.float64)
            if a.ndim < 2:
                continue
            span = float(np.max(a.max(axis=-2) - a.min(axis=-2)))
            amax = float(np.abs(a).max())
            d0_str = ""
            cur = state.get(key)
            if cur is not None:
                c = np.asarray(cur, dtype=np.float64)
                # state is (N, T, D); take the latest frame -> (N, D)
                c0 = c[..., -1, :] if c.ndim >= 2 else c
                first = a[..., 0, :]  # first predicted step -> (N, D)
                if c0.shape[-1] == first.shape[-1]:
                    d0_str = f" d0={float(np.abs(first - c0).max()):.4f}"
            parts.append(f"{key}[span={span:.4f}{d0_str} amax={amax:.3f}]")
        print("[action-values] " + "  ".join(parts), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint-* dir")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()

    # Serve the raw nested Gr00tN2Policy: replace the flat adapter with our
    # nested passthrough + WBC-zero adapter before main() constructs the server.
    rgs.Gr00tFlatPolicyWrapper = NestedNavAdapter

    config = rgs.ServerConfig(
        embodiment_tag=schema.EmbodimentTag(args.embodiment_tag),
        model_path=args.checkpoint,
        host=args.host,
        port=args.port,
        num_inference_timesteps=args.denoising_steps,
    )
    rgs.main(config)


if __name__ == "__main__":
    main()
