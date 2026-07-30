# ReMind inference examples

The seven original project-page presets are **all I2V**, even when their
control is an occluder or camera trajectory:

1. latte-art occluder recovery;
2. pancake-batter occluder recovery;
3. whisking with camera pan;
4. granular pouring with camera pan;
5. clean dog-motion I2V;
6. clean dry-ice/foam I2V; and
7. clean flour-pour I2V.

Four additional presets are explicitly **V2V**:

8. syrup decoration on a frosted slice with occluder recovery V2V;
9. flaming-drink occluder recovery V2V;
10. carbonated-drink pouring with camera pan V2V; and
11. viscous yellow slime accumulating with camera pan V2V.

There are intentionally no clean V2V examples and none of the V2V scenes
duplicates the seven I2V presets. Each V2V preset consumes only its bundled
21-frame source-reference prefix, corresponding to two 3-frame latent chunks
(6 latent frames); it never reads a full reference video or future frames.

Every preset fixes the 832×480 input, caption, seed, and four-step DMD-EMA
rollout. Both the I2V and V2V camera pairs bundle the same exact pair-fixed
InSpatio `camera.npz` trajectories used for their page videos. The occluder
pairs preserve the exact enter/hold/exit frames and appearance text; the
occluder itself is described through chunk-local text and is not supplied as
visual conditioning.

Run one case with the command in the root README. Use `--dry-run` to inspect
its seven chunk prompts and camera metrics without loading model weights.
See [`../docs/inference.md`](../docs/inference.md) before writing a custom
caption or changing a control schedule.
