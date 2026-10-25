"""Gymnasium environment for Subway Surfers running in BlueStacks.

    from qtrain.env import SubwaySurfersEnv
    env = SubwaySurfersEnv(serial="127.0.0.1:5556", window="BlueStacks App Player 1")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Observation: (frame_stack, 84, 84) uint8 grayscale frames, oldest first.
Actions:     0 noop, 1 left, 2 right, 3 up (jump), 4 down (roll).
Reward:      +alive_reward per step survived, crash_penalty when the run ends.
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .adb import AdbDevice, best_backend
from .capture import make_capture
from .controller import ACTIONS, ControllerConfig, GameController
from .screens import Calibration


class SubwaySurfersEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, serial: str = "127.0.0.1:5556", window: str = "BlueStacks App Player",
                 calibration_dir: str = "assets", capture: str = "dxcam",
                 config: ControllerConfig | None = None, render_mode: str | None = None):
        cfg = config or ControllerConfig()
        self.metadata = {**self.metadata, "render_fps": cfg.hz}
        self.render_mode = render_mode
        self.device = AdbDevice(serial, backend=best_backend(serial))
        print(f"[qtrain] {serial}: screen {self.device.width}x{self.device.height}, "
              f"input backend '{self.device.backend}'")
        self.capture = make_capture(capture, window=window, serial=serial)
        self.ctrl = GameController(self.device, self.capture, Calibration.load(calibration_dir), cfg)
        self.observation_space = spaces.Box(0, 255, (cfg.frame_stack, cfg.obs_size, cfg.obs_size), np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self.ctrl.reset()

    def step(self, action):
        return self.ctrl.step(int(action))

    def render(self):
        if self.ctrl.last_canon is None:
            return None
        return cv2.cvtColor(self.ctrl.last_canon, cv2.COLOR_BGR2RGB)

    def close(self):
        self.device.close()
        self.capture.close()
