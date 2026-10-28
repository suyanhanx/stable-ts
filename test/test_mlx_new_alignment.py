"""No model downloads: compare MLX matrices against the upstream Torch algorithm.

Run on Apple Silicon: python -m unittest discover -s test -p test_mlx_new_alignment.py
"""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

HAS_MLX = importlib.util.find_spec('mlx') is not None and importlib.util.find_spec('mlx_whisper') is not None
if HAS_MLX:
    import mlx.core as mx
    from stable_whisper import mlx_timing
    from stable_whisper.alignment import get_whisper_alignment_func
    from stable_whisper.options import AllOptions
    from stable_whisper.timing import _compute_atten_weights_new
    from whisper.timing import dtw as torch_dtw
    from mlx_whisper.timing import dtw as mlx_dtw


@unittest.skipUnless(HAS_MLX, 'requires MLX and mlx-whisper on Apple Silicon')
class NewMLXAlignmentTests(unittest.TestCase):
    def test_matrix_and_path_match_upstream(self):
        rng = np.random.default_rng(21)
        # Non-tied head scores; include context and unused padded frames.
        raw = [rng.normal(size=(1, 4, 12, 29)).astype(np.float32) for _ in range(3)]
        tokenizer = SimpleNamespace(sot_sequence=[1, 2, 3])
        for settings in ({'topk': 5}, {'topk': 9, 'w_coverage': 0.7},
                         {'topk': 3, 'w_colnorm': 0, 'w_rownorm': 2},
                         {'topk': 12, 'w_colnorm': 2, 'w_rownorm': 0}):
            with self.subTest(settings=settings):
                expected = _compute_atten_weights_new(
                    model=None, tokenizer=tokenizer, text_tokens=[], mel=None,
                    num_samples=23 * 320, tokens=None,
                    cache={'qks': [torch.from_numpy(x) for x in raw]}, **settings
                ).numpy()
                actual = np.array(mlx_timing.alignment_matrix(
                    [mx.array(x) for x in raw], 3, 23, **settings))
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                for a, b in zip(mlx_dtw(-actual), torch_dtw(torch.from_numpy(-expected))):
                    np.testing.assert_array_equal(a, b)

    def test_filter_matches_torch_including_short_audio(self):
        from whisper.timing import median_filter
        rng = np.random.default_rng(9)
        for length in (2, 4, 19):
            raw = rng.normal(size=(3, 9, length)).astype(np.float32)
            actual = np.array(mlx_timing._median_filter(mx.array(raw), 7))
            np.testing.assert_array_equal(actual, median_filter(torch.from_numpy(raw), 7).numpy())

    def test_options_rejected_instead_of_ignored(self):
        for value in ('typo', {'top_k': 3}, {'topk': 0}, {'topk': True},
                      {'w_colnorm': float('nan')}, {'w_coverage': -1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mlx_timing.validate_options(value)
        for options in ({'dynamic_heads': True}, {'extra_models': [object()]}):
            with self.assertRaises(NotImplementedError):
                mlx_timing.validate_options('new', **options)
        with self.assertRaises(ValueError):
            mlx_timing.alignment_matrix([mx.zeros((1, 2, 8, 10))], 3, 10, topk=3)

    def test_factory_routes_legacy_and_new_without_global_patching(self):
        # Verify the public factory forwards options and leaves upstream legacy
        # processing intact, using only a short generated waveform.
        model = SimpleNamespace(dims=SimpleNamespace(n_mels=80))
        tokens = [SimpleNamespace(tokens=[10], word='test')]
        for aligner in ('legacy', 'new', {'topk': 2, 'w_coverage': 0.5}):
            with self.subTest(aligner=aligner):
                options = AllOptions({'aligner': aligner}, vanilla_align=True)
                target = ('mlx_whisper.timing.add_word_timestamps' if aligner == 'legacy'
                          else 'stable_whisper.mlx_timing.add_word_timestamps')
                with patch(target) as invoke:
                    compute = get_whisper_alignment_func(model, None, 'mlx', options)
                    compute(torch.zeros(16000), tokens)
                    invoke.assert_called_once()
                    if isinstance(aligner, dict):
                        self.assertEqual(invoke.call_args.kwargs['topk'], 2)
                        self.assertEqual(invoke.call_args.kwargs['w_coverage'], 0.5)


if __name__ == '__main__':
    unittest.main()
