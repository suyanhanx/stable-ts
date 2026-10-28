# New alignment algorithm on MLX

This fork adds opt-in `aligner="new"` support to `align()` and `align_words()`
for models loaded with `load_mlx_whisper()`. It ports stable-ts' upstream
`_compute_atten_weights_new()` algorithm, based on
[Whisper character alignment](https://arxiv.org/abs/2509.09987).

```python
import stable_whisper
import mlx.core as mx

model = stable_whisper.load_mlx_whisper("/path/to/local/whisper-large-v3", dtype=mx.float16)
result = model.align(
    "audio.wav", "校正済みの日本語原稿。", language="ja",
    aligner="new", regroup=False,
)
result.save_as_json("alignment-new.json")
```

Install this checkout with the MLX extra in a separate environment to try the
fork: `python -m pip install '.[mlx]'`. Existing installed environments are not
modified merely by cloning this repository.

`aligner="legacy"` remains the default and delegates to mlx-whisper unchanged.
To configure the new algorithm, pass a dictionary instead of `"new"`:

```python
result = model.align(
    "audio.wav", "校正済みの日本語原稿。", language="ja",
    aligner=dict(topk=20, w_colnorm=1, w_rownorm=1, w_coverage=0),
)
```

Unknown options and unsupported `dynamic_heads` / `extra_models` raise an
error on the MLX alignment path. Dynamic head selection is a distinct legacy
algorithm and is not implemented by this port. Transcription and `refine()`
are not changed by this patch.

## Implementation

- Reuse MLX Whisper's forward pass and cross-attention logits.
- Median-filter raw attention logits, then apply softmax; this order differs
  from the legacy algorithm.
- Score heads with column/row norms and the optional coverage penalty.
- Keep the global top-k heads while evaluating one layer at a time. Temporary
  filtered matrices do not accumulate for all layers.
- Normalize selected heads by column norm and average; use float32 for scoring
  and normalization, even when model inference uses float16.
- Run the existing CPU DTW and retain mlx-whisper's punctuation and segment
  boundary postprocessing (attributed in `mlx_timing.py`). No global function
  patching or model weight changes are needed.

For tied head scores, the MLX implementation retains earlier heads. PyTorch's
`topk` does not promise the same tie ordering. Floating-point model inference
and attention precision can also change near-tied selections; bit-identical
timestamps across model backends are not promised.

## Verification

On Apple Silicon with the MLX extra installed:

```sh
python -m unittest discover -s test -p test_mlx_new_alignment.py -v
```

The tests require no model downloads. They compare the actual upstream Torch
algorithm and MLX implementation on shared attention inputs, check DTW paths,
test temporal filtering and verify option dispatch/rejection.

Algorithm parity does not establish Japanese subtitle accuracy. Compare the
same audio and corrected transcript under both algorithms, retaining VAD and
segmentation settings. Use manually checked speech boundaries for accuracy;
text coverage and absence of zero-duration units are structural checks only.
