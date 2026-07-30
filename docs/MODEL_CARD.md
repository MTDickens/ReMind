# ReMind model card

ReMind post-trains causal Wan video generators to preserve and evolve hidden
state through camera motion, occlusion, and illumination changes. This release
contains inference code for the models described in
“Teaching Video Generators to Remember: Eliciting Dynamic Memory for
Out-of-Sight State Evolution.”

## Model variants

| Variant | Upstream base | Resolution | ReMind checkpoint | DMD EMA LoRA |
|---|---|---:|---|---|
| ReMind 5B | Wan2.2-TI2V-5B | 832×480 | `ReMind-5B.safetensors` | `ReMind-5b-dmd-ema.safetensors` |
| ReMind 1.3B | Wan2.1-T2V-1.3B | 832×480 | Coming soon | Coming soon |

## Released checkpoint contents

The initial Hugging Face release contains the two validated ReMind 5B
inference artifacts directly at the repository root:

```text
ReMind-5B.safetensors
ReMind-5b-dmd-ema.safetensors
```

`ReMind-5B.safetensors` is the complete ReMind generator.
`ReMind-5b-dmd-ema.safetensors` is the EMA student LoRA produced by SF-DMD
post-training; it is not a standalone checkpoint and must be applied to the
ReMind-5B generator.

ReMind 1.3B weights are not part of the initial release and remain coming soon.

The inference load order is:

1. load the official Wan 2.2 TI2V 5B model directory for its architecture,
   text encoder, tokenizer, and VAE;
2. load `ReMind-5B.safetensors`; and
3. apply `ReMind-5b-dmd-ema.safetensors`.

Do not apply the DMD EMA LoRA directly to the unmodified upstream Wan generator
or mix it with another base checkpoint. Raw student, critic,
optimizer, and training-state checkpoints are training artifacts and are not
part of the inference release.

## Intended use

The models are research artifacts for studying dynamic memory, controlled
camera motion, reversible visibility disturbances, and clean image-to-video
generation. The included presets demonstrate representative input and control
formats; they are not a benchmark or a guarantee of behavior.

## Limitations

- Long human motion can accumulate autoregressive error and lose visual
  quality.
- Camera control is approximate and may not follow every requested trajectory.
- Recovery depends on scene content, event timing, seed, and model size.
- Generated videos can contain physical, geometric, identity, lighting, and
  temporal artifacts.
- The models should not be used for high-stakes decisions or to misrepresent
  generated media as real.

## License and release status

No weights are stored in this Git repository. The released ReMind 5B
ReMind-5B and SF-DMD/EMA checkpoints are licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) as specified in
`WEIGHTS_LICENSE.md`. ReMind 1.3B weights are coming soon. The public Hugging
Face model repository is available at
[AppliedIntuitionResearch/ReMind](https://huggingface.co/AppliedIntuitionResearch/ReMind).

Code and third-party terms are summarized in the root `../README.md`,
`../LICENSE`, and `../NOTICE` files and in `THIRD_PARTY_NOTICES.md`.

## Citation

See the BibTeX entry at the bottom of the root `../README.md`.
