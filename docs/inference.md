# ReMind inference and caption guide

This guide describes the conditioning contract used by the public presets and
project-page examples. Start from the closest preset in `examples/presets/`
rather than building a case from an unrelated inference wrapper.

## Canonical release recipe

Keep these settings for page-aligned 5B DMD-EMA inference:

- 81 output frames at 832×480 and 16 fps;
- 21 latent frames with `chunk_size: 3`, giving seven prompt chunks;
- four shifted denoising steps: `[1000, 937, 833, 625]`; and
- the ReMind-5B generator followed by the DMD EMA adapter.

I2V presets use exactly one input image and one seed latent frame. V2V presets
use only a 21-frame video prefix (`seed_latent_frames: 6`); they must not read
future frames or a complete reference video.

## Base caption contract

The preset's `prompt` is always a **content-only temporal caption**. Describe
what happens in the scene, not how the camera or visibility control behaves.
For a curated case, use roughly 60–170 words and cover:

1. the initial objects and physical state;
2. the action and visible state changes;
3. a meaningful middle-stage evolution;
4. the final state; and
5. the governing dynamics when useful, such as flow, spreading, impact,
   deformation, accumulation, mixing, melting, or human/animal motion.

Temporal anchors such as “At the start,” “As … continues,” “Midway,” and “By
the end” are useful. Preserve object identity and describe one continuous
evolution instead of listing disconnected frames.

Do **not** put any of the following in the base caption:

- camera instructions such as “camera pans,” “pan left/right,” “tilt,”
  “orbit,” “zoom,” or “returns to the original view”;
- visibility-control instructions such as “an occluder enters,” “the scene is
  blocked,” or “the object is revealed again”;
- synthetic light-toggle instructions such as “the lights turn off/on”; or
- generation instructions such as “the model should remember,” “recover the
  scene,” or “continue during the hidden interval.”

Camera geometry, occlusion timing, and light timing have separate conditioning
channels. Mixing them into `prompt` duplicates or conflicts with those
controls.

### Aligned caption example

> A steady stream of small brown seeds is poured onto a shallow plate, building
> a growing conical heap at center. At the start a thin layer is spread across
> the plate. As pouring continues, falling grains strike the pile, bounce, and
> slide outward while the mound becomes taller and denser. Midway, impacts send
> individual seeds rolling down the slope and widening the base. By the end,
> the central cone is pronounced and scattered grains surround it. The motion
> shows continuous inflow, collision, compaction, and granular avalanching.

The same caption can be used for clean I2V or a bundled camera trajectory; do
not append camera wording.

## Caption behavior by control mode

| `control.kind` | Base `prompt` | Additional conditioning |
|---|---|---|
| `clean` | Dynamics only | Static camera; the same caption is used across all seven chunks. |
| `camera` | Dynamics only; no camera words | Set `control.trajectory` to a bundled pair-fixed `.npz`; PM-RoPE carries the camera motion. |
| `occluder` | Dynamics only; no occluder/recovery words | `control.params` and `control.meta` generate chunk-local enter/hold/exit prompts. |
| `light` | Dynamics only; no light-toggle/recovery words | Light parameters generate chunk-local off/hold/on prompts. |

For occluder and light cases, the resolved seven-chunk schedule follows this
structure:

- chunks 0–1: content-only action;
- chunk 2: the occluder enters or the lights turn off progressively;
- chunk 3: the scene is fully blocked or dark while the action continues;
- chunk 4: the occluder exits or the lights turn back on;
- chunks 5–6: visibility is restored and the action continues without reset.

Keep this lifecycle out of the base `prompt`; `inference.py` constructs the
chunk-local text internally. The canonical I2V occluder timing is enter
frames 24→30, hold 30→42, and exit 42→48. For V2V, copy an existing V2V preset
because the event is placed after its conditioned prefix.

## Camera trajectories

For curated or project-page-aligned cases, use the exact pair-fixed InSpatio
trajectories bundled under `examples/assets/*_camera.npz`. Preserve the NPZ,
including its intrinsics: do not rescale, normalize, recenter, smooth,
transpose, or replace it with an analytic yaw/pitch curve.

The inference code supports an analytic camera fallback, but it is not the
project-page recipe and should not be used to claim page-aligned reproduction.
With a bundled trajectory, the dynamics-only caption is repeated for all seven
chunks and camera motion is supplied only through PM-RoPE.

## Preparing a custom preset

1. Copy the closest I2V or V2V YAML from `examples/presets/`.
2. Change `name`, the input path, `prompt`, and `seed`; keep the canonical
   frame, resolution, fps, and chunk settings.
3. For camera control, point `control.trajectory` to a bundled pair-fixed NPZ.
   For an occluder, copy the explicit `params`/`meta` structure from an
   occluder preset and adjust only the intended appearance and timing.
4. Validate the resolved schedule before loading weights:

```bash
python inference.py --preset /path/to/custom_case.yaml --dry-run
```

The dry run should report 21 latent frames, seven prompt entries, and the
expected control kind. A camera case should also report nonzero camera metrics.
Then run the checkpoint command from the root `README.md`.

## Common caption mistakes

| Avoid | Use instead |
|---|---|
| “The camera pans left while batter is whisked.” | Describe only how the batter and whisk evolve; provide the NPZ separately. |
| “A box blocks the scene, then leaves to reveal the finished pour.” | Describe the uninterrupted pour; put the box lifecycle in `control.params`. |
| “The lights turn off while the material keeps melting.” | Describe the melting process; use `control.kind: light`. |
| A short object list with no temporal change | State the initial, middle, and final physical states explicitly. |

See `examples/README.md` for the released I2V/V2V case inventory and exact
project-page inputs.
