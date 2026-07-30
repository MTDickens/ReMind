# ReMind dataset card

The planned ReMind dataset release contains approximately 1.2 million video
examples assembled for dynamic-memory training. It combines synthetic camera
motion derived from InSpatio-World with strong temporal dynamics and reversible
visibility disturbances derived from Helios.

## Contents

- synthetic camera pan, return, and out-of-sight state-evolution examples;
- strong temporal dynamics such as pouring, melting, mixing, and motion by
  people and animals;
- reversible occluder and illumination-toggle events; and
- clean videos used to retain general image-to-video capability.

InSpatio camera-augmentation records must use the corrected pairing between
video, caption, identity, and camera metadata. The earlier mispaired manifests
are not part of this release.

## Manifest format

Training consumes JSONL or JSONL.GZ. A minimal public row is:

```json
{"video":"videos/example.mp4","caption":"A person pours water into a glass.","source":"helios","id":"example","num_frames":81}
```

Paths in the public manifests must be relative paths or public URLs. Internal
storage locations, workload identifiers, private validation manifests, and raw
credentials are excluded.

## Intended use and limitations

The dataset is intended for research on video generation, temporal dynamics,
camera control, and hidden-state recovery. Synthetic captions and metadata may
be incomplete or incorrect, and the source distribution does not represent all
people, activities, environments, or camera styles.

## License and release status

The public dataset repository is available at
[AppliedIntuitionResearch/ReMind1M](https://huggingface.co/datasets/AppliedIntuitionResearch/ReMind1M).
Released ReMind dataset artifacts are licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); see the repository
license for details.
