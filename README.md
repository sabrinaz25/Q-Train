# Q-Train 🚆

An agent that learns to play **Subway Surfers** from raw screen pixels, running
across parallel Android emulator instances on a single Windows laptop.

> 🚧 Work in progress. Current stage: **2 – screen recognition & hands-free episodes**

The hard part of playing a real mobile game isn't the RL algorithm. It's the
plumbing: getting frames out of the emulator and swipes back in fast enough to
react at game speed, detecting crashes reliably from pixels, and collecting
enough experience when every episode runs in real time. Q-Train treats those as
first-class engineering problems and measures everything.

## Roadmap

| # | Stage | Status |
|---|---|---|
| 1 | Capture + input backends, latency benchmark | 🔁 capture ✅, fast input in progress |
| 2 | Screen recognition (menu / run / "Save me?" / popups) + automatic restarts, wrapped as a Gymnasium env | 🔨 in progress |
| 3 | Scale-out: N emulator instances as one vectorized env; throughput vs. N | ⏳ |
| 4 | Record human gameplay → behavior-cloning CNN (trained on Colab GPU) | ⏳ |
| 5 | RL fine-tuning across the parallel envs; ONNX inference on the laptop | ⏳ |

## Results

### Stage 1: latency
Measured on an HP Spectre 16 (i7-1360P, 16 GB, Windows 11), BlueStacks 5, Android 11 instance.

| Input backend | Round-trip p50 | p95 |
|---|---|---|
| `spawn`: new `adb shell input swipe` per action | 623 ms | 660 ms |
| `shell`: one persistent `adb shell` | 572 ms | 601 ms |
| ~~`cmd`: persistent shell + `cmd input`~~ | ~~10 ms~~ | invalid: see below |

| Capture backend | Per frame p50 | Frames/s |
|---|---|---|
| `adb exec-out screencap` | 619–908 ms | ~1 |
| `mss` (GDI window grab) | 16.9 ms | 58 |
| **`dxcam`** (Desktop Duplication, time to a *new* frame) | **16.7 ms** | **60** (the game's frame rate) |

**Findings**
- **Input is the bottleneck: ~600 ms per action.** Keeping one ADB shell open only saved
  ~50 ms; the real cost is Android's `input` command starting a fresh runtime on every
  swipe. That's far too slow to play at game speed (target: 15 actions/s).
- **A false result, caught.** `cmd input` first measured 10 ms (a "60× speedup"), but on
  this BlueStacks Android 11 build it replies `No shell command implementation.` and does
  nothing: the benchmark was timing an error message. Lesson: latency numbers need
  end-to-end verification that the action actually happened. Next step: an end-to-end
  check (input → visible on screen) and a persistent on-device input injector.
- **Capture keeps up with the game.** Both desktop-capture backends deliver every frame
  the game renders (60 fps); screenshots over ADB are ~50× too slow to play with.
- **A benchmark bug worth noting:** dxcam first reported 0.07 ms/frame because it returns
  nothing when the screen hasn't changed and the code re-used the previous frame.
  Measuring *time to a new frame* gives the honest number.

Raw numbers: [`results/`](results/).

## Setup (Windows)

1. **Emulator:** install [BlueStacks 5](https://www.bluestacks.com/). In the
   Multi-Instance Manager, create an **Android 11** instance. Install Subway Surfers
   from the Play Store inside it.
2. **Enable ADB:** BlueStacks *Settings → Advanced → Android Debug Bridge (ADB)*
   → on. Note the address it shows (e.g. `127.0.0.1:5556`; each instance gets its own port).
3. **adb:** install [Android SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools),
   unzip it, and add the folder to your PATH (or set `QTRAIN_ADB` to the full path of `adb.exe`).
4. **Python 3.11:**
   ```powershell
   conda create -n qtrain python=3.11 -y
   conda activate qtrain
   pip install -r requirements-dev.txt
   ```

## Stage 1: benchmark

Open Subway Surfers to its main menu, keep the BlueStacks window visible
(not minimized or covered), then:

```powershell
python -m qtrain.bench --list-devices       # e.g. 127.0.0.1:5556
python -m qtrain.bench --list-windows       # find the emulator window title
python -m qtrain.bench --serial 127.0.0.1:5556 --window "BlueStacks App Player 1" --save-frames
```

| Capture backend | How it works |
|---|---|
| `adb` | `adb exec-out screencap`: device reads back its framebuffer and ships it over ADB |
| `mss` | GDI screenshot of the emulator window |
| `dxcam` | Windows Desktop Duplication API |

| Input backend | How it works |
|---|---|
| `spawn` | new `adb shell input swipe` process per action (the naive approach) |
| `shell` | one persistent `adb shell`; commands streamed over stdin |
| `cmd` | persistent shell + `cmd input`, skipping the `input` wrapper (only where Android implements it; auto-detected) |

## Stage 2: playing hands-free

The environment has to get from *whatever is on screen* into a running game,
and notice when the run ends, with nobody clicking. How it sees the game:

- **Game area.** The BlueStacks window is cropped to just the game and resized to a fixed
  360×640 "canonical" image, so moving or resizing the window doesn't break anything.
- **Known screens** (menu, run in progress, "Save me?", score screen, any popups you
  teach it) are recognised by small templates matched near where they were calibrated
  (< 1 ms each). Each has a role (`start`, `playing`, `crash`, `dismiss`) and a tap point.
- **Unknown popups with a red X** are closed automatically. The detector finds compact
  red blobs, then checks that the white glyph inside is an X: it must cover both diagonals
  and leave the four wedges between its arms empty. That second test is what stops it
  tapping the red *notification badges* ("1", "2", "3"…) all over the main menu. It's
  unit-tested against every digit at four badge sizes and gives zero false positives on
  a real menu screenshot.
- **Anything else:** wait, then press Android Back.

### 1. Calibrate (once)
Put this terminal and BlueStacks side by side:
```powershell
python -m qtrain.calibrate --window "BlueStacks App Player 1"
```
For each screen it asks you to bring the game there, drag a box around something unique to
that screen, and click where to tap. It ends with a cross-check table and warns if a template
could be confused with another screen. Re-do one screen with `--only save_me`; teach it a
recurring popup with `--add-popup daily_reward`. Calibration lives in `assets/`, which is
gitignored because the templates are crops of the game's artwork.

### 2. Watch what it sees
```powershell
python -m qtrain.watch --window "BlueStacks App Player 1"
```
A live window showing the detected screen, template scores and any red X's (boxed in pink).
Click through the game's menus and check every screen is recognised. `s` saves a frame, `q` quits.

### 3. Play whole episodes
```powershell
python -m qtrain.run --serial 127.0.0.1:5556 --window "BlueStacks App Player 1" --episodes 5
```
A random agent plays 5 runs back to back (reset → play → crash → reset) hands-free. It
reports steps per episode, reset time, achieved control rate and the share of wall-clock
time spent actually playing, and saves it all to `results/`.

### From Python
```python
from qtrain.env import SubwaySurfersEnv
env = SubwaySurfersEnv(serial="127.0.0.1:5556", window="BlueStacks App Player 1")
obs, info = env.reset()          # (4, 84, 84) uint8 stacked grayscale frames
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```
Actions: `0` noop, `1` left, `2` right, `3` jump, `4` roll. Reward: +0.1 per step
survived, −1 on crash. It never taps the "Save me" button, so no keys are spent.

Known limitation: the "Save me?" prompt appears about a second after the actual
collision, so the crash penalty lands a little after the mistake.

## Project layout

```
qtrain/
  adb.py         # device connection, swipe geometry, input backends
  capture.py     # DPI-aware window lookup + capture backends
  bench.py       # latency benchmark CLI
  vision.py      # game-area crop, observations, red-X popup detector
  screens.py     # screen templates, calibration file, classifier
  controller.py  # reset/step game loop (no Gymnasium; testable with fakes)
  env.py         # Gymnasium environment
  calibrate.py   # interactive calibration
  watch.py       # live debug view
  run.py         # play episodes with a baseline policy
tests/           # unit tests incl. a scripted fake game (CI on Ubuntu + Windows)
results/         # committed benchmark and run outputs
```

## Notes

- No game assets, APKs, or recorded gameplay are committed; they're not ours to redistribute.
- Runs offline against a local emulator only; never against leaderboards or online events.
