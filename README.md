# Q-Train 🚆

An agent that learns to play **Subway Surfers** from raw screen pixels, running
across parallel Android emulator instances on a single Windows laptop.

> 🚧 Work in progress. Current stage: **1 – low-latency capture & input pipeline**

The hard part of playing a real mobile game isn't the RL algorithm. It's the
plumbing: getting frames out of the emulator and swipes back in fast enough to
react at game speed, detecting crashes reliably from pixels, and collecting
enough experience when every episode runs in real time. Q-Train treats those as
first-class engineering problems and measures everything.

## Roadmap

| # | Stage | Status |
|---|---|---|
| 1 | Capture + input backends, latency benchmark | 🔨 in progress |
| 2 | Game-state detection (game over / "Save me?") + automatic restarts, wrapped as a Gymnasium env | ⏳ |
| 3 | Scale-out: N emulator instances as one vectorized env; throughput vs. N | ⏳ |
| 4 | Record human gameplay → behavior-cloning CNN (trained on Colab GPU) | ⏳ |
| 5 | RL fine-tuning across the parallel envs; ONNX inference on the laptop | ⏳ |

## Results

### Stage 1: latency
_Run `python -m qtrain.bench` and paste the table here._

## Setup (Windows)

1. **Emulator:** install [BlueStacks 5](https://www.bluestacks.com/). In the
   Multi-Instance Manager, create an **Android 11** instance (needed for the
   fastest input backend). Install Subway Surfers from the Play Store inside it.
2. **Enable ADB:** BlueStacks *Settings → Advanced → Android Debug Bridge (ADB)*
   → on. Note the address it shows (usually `127.0.0.1:5555`).
3. **adb:** install [Android SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools),
   unzip it, and add the folder to your PATH (or set `QTRAIN_ADB` to the full path of `adb.exe`).
4. **Python 3.11:**
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements-dev.txt
   ```

## Running the benchmark

Open Subway Surfers to its main menu, keep the BlueStacks window visible
(not minimized or covered), then:

```powershell
python -m qtrain.bench --list-devices       # should show 127.0.0.1:5555
python -m qtrain.bench --list-windows       # find the emulator window title
python -m qtrain.bench --serial 127.0.0.1:5555 --window "BlueStacks App Player" --save-frames
```

It measures every backend and saves raw numbers to `results/`:

| Capture backend | How it works |
|---|---|
| `adb` | `adb exec-out screencap`: device reads back its framebuffer and ships it over ADB |
| `mss` | GDI screenshot of the emulator window |
| `dxcam` | Windows Desktop Duplication API |

| Input backend | How it works |
|---|---|
| `spawn` | new `adb shell input swipe` process per action (the naive approach) |
| `shell` | one persistent `adb shell`; commands streamed over stdin |
| `cmd` | persistent shell + `cmd input`, skipping the `input` wrapper (Android 10+) |

`--save-frames` writes one sample frame per backend to `data/` (gitignored) so
you can check the capture region is right.

## Project layout

```
qtrain/
  adb.py       # device connection, swipe geometry, input backends
  capture.py   # DPI-aware window lookup + capture backends
  bench.py     # latency benchmark CLI
tests/         # unit tests (run in CI on Ubuntu + Windows)
results/       # committed benchmark outputs
```

## Notes

- No game assets, APKs, or recorded gameplay are committed; they're not ours to redistribute.
- Runs offline against a local emulator only; never against leaderboards or online events.
