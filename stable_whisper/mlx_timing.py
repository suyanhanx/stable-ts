"""MLX implementation of stable-ts' new attention-head alignment algorithm.

The segment postprocessing below is adapted from mlx-whisper's timing.py.
Copyright (c) 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import itertools
import math

import mlx.core as mx
import numpy as np
from mlx_whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from mlx_whisper.timing import WordTiming, dtw, merge_punctuations


def validate_options(aligner, dynamic_heads=None, extra_models=None):
    if dynamic_heads or extra_models:
        raise NotImplementedError(
            'MLX alignment does not support dynamic_heads or extra_models; '
            'use aligner="new" for the new head-selection algorithm.'
        )
    if isinstance(aligner, str) and aligner in ('legacy', 'new'):
        return {} if aligner == 'new' else None
    if not isinstance(aligner, dict):
        raise ValueError('MLX aligner must be "legacy", "new", or a dictionary of new-aligner options')
    allowed = {'topk', 'w_colnorm', 'w_rownorm', 'w_coverage'}
    if set(aligner) - allowed:
        raise ValueError(f'Unknown MLX aligner options: {sorted(set(aligner) - allowed)}')
    topk = aligner.get('topk', 20)
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError('topk must be a positive integer')
    for name in allowed - {'topk'}:
        value = aligner.get(name, 0 if name == 'w_coverage' else 1)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')
    return dict(aligner)


def _median_filter(x, width):
    """Reflect-padded temporal median, matching whisper.timing.median_filter."""
    if width < 1 or width % 2 != 1:
        raise ValueError('Median filter width must be a positive odd integer')
    pad = width // 2
    if pad == 0 or x.shape[-1] <= pad:
        return x
    padded = mx.concatenate((x[..., 1:pad + 1][..., ::-1], x,
                             x[..., -pad - 1:-1][..., ::-1]), axis=-1)
    windows = mx.stack([padded[..., i:i + x.shape[-1]] for i in range(width)], axis=-1)
    return mx.sort(windows, axis=-1)[..., pad]


def alignment_matrix(cross_qk, sot_length, num_frames, *, topk=20,
                     w_colnorm=1, w_rownorm=1, w_coverage=0,
                     medfilt_width=7, qk_scale=1.0):
    """Return the token/time matrix; keep at most topk candidate heads per layer.

    Filtering precedes softmax, unlike the legacy MLX algorithm. All scoring
    and normalization use float32 on the selected MLX device. CPU reads contain
    only the small head-score vectors. DTW runs separately on the final matrix.
    """
    validate_options(dict(topk=topk, w_colnorm=w_colnorm,
                          w_rownorm=w_rownorm, w_coverage=w_coverage))
    if not cross_qk or num_frames < 1:
        raise ValueError('Alignment requires attention heads and at least one audio frame')
    if topk > sum(qk.shape[1] for qk in cross_qk):
        raise ValueError('topk exceeds the number of available attention heads')
    candidates = None
    candidate_scores = np.empty(0, dtype=np.float32)
    for layer in cross_qk:
        weights = _median_filter(layer[0, :, :, :num_frames].astype(mx.float32), medfilt_width)
        weights = mx.softmax(weights * qk_scale, axis=-1, precise=True)
        scores = mx.zeros((weights.shape[0],), dtype=mx.float32)
        if w_colnorm > 0:
            scores = scores + w_colnorm * mx.linalg.norm(weights, axis=-2).sum(-1)
        if w_rownorm > 0:
            scores = scores + w_rownorm * mx.linalg.norm(weights, axis=-1).sum(-1)
        if w_coverage > 0:
            coverage = weights.sum(axis=-2)
            penalty = mx.maximum(coverage, 0.5).sum(-1) - coverage.shape[-1] * 0.5
            scores = scores - w_coverage * penalty
        # Bound the live temporary graph and candidates rather than stacking
        # every filtered layer of large-v3 at once.
        mx.eval(weights, scores)
        score_values = np.array(scores)
        if not np.isfinite(score_values).all():
            raise ValueError('Non-finite attention scores; cannot align this window')
        local = np.argsort(-score_values, kind='stable')[:topk]
        selected = weights[mx.array(local.astype(np.int32))]
        combined = selected if candidates is None else mx.concatenate((candidates, selected), axis=0)
        combined_scores = np.concatenate((candidate_scores, score_values[local]))
        keep = np.argsort(-combined_scores, kind='stable')[:topk]
        candidates = combined[mx.array(keep.astype(np.int32))]
        candidate_scores = combined_scores[keep]
        mx.eval(candidates)
    norms = mx.linalg.norm(candidates, axis=-2, keepdims=True)
    matrix = mx.mean(candidates / norms, axis=0)[sot_length:-1]
    mx.eval(matrix)
    if not bool(mx.all(mx.isfinite(matrix)).item()):
        raise ValueError('Non-finite alignment matrix; cannot align this window')
    return matrix


def find_alignment(model, tokenizer, text_tokens, mel, num_frames, **options):
    if not text_tokens:
        return []
    tokens = mx.array([*tokenizer.sot_sequence, tokenizer.no_timestamps,
                       *text_tokens, tokenizer.eot])
    logits, cross_qk = model.forward_with_cross_qk(mel[None, :], tokens[None, :])
    sampled_logits = logits[0][len(tokenizer.sot_sequence):-2, :tokenizer.eot]
    probabilities = mx.softmax(sampled_logits, precise=True, axis=-1)
    text_probabilities = np.array(mx.take_along_axis(
        probabilities, mx.array(text_tokens)[:, None], axis=1).squeeze(1))
    matrix = alignment_matrix(cross_qk, len(tokenizer.sot_sequence), num_frames // 2, **options)
    text_indices, time_indices = dtw(-np.array(matrix))
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        return []
    boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    return [WordTiming(word, word_token, jump_times[i], jump_times[j],
                       float(np.mean(text_probabilities[i:j])))
            for word, word_token, i, j in zip(words, word_tokens, boundaries[:-1], boundaries[1:])]


def add_word_timestamps(
    *,
    segments: list,
    model,
    tokenizer,
    mel: mx.array,
    num_frames: int,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp: float,
    **kwargs,
):
    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)
    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    # a better segmentation algorithm based on VAD should be able to replace this.
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=float(timing.probability),
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
            # ensure the first and second word after a pause is not longer than
            # twice the median word duration.
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (
                    len(words) > 1
                    and words[1]["end"] - words[0]["start"] > max_duration * 2
                )
            ):
                if (
                    len(words) > 1
                    and words[1]["end"] - words[1]["start"] > max_duration
                ):
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # prefer the segment-level start timestamp if the first word is too long.
            if (
                segment["start"] < words[0]["end"]
                and segment["start"] - 0.5 > words[0]["start"]
            ):
                words[0]["start"] = max(
                    0, min(words[0]["end"] - median_duration, segment["start"])
                )
            else:
                segment["start"] = words[0]["start"]

            # prefer the segment-level end timestamp if the last word is too long.
            if (
                segment["end"] > words[-1]["start"]
                and segment["end"] + 0.5 < words[-1]["end"]
            ):
                words[-1]["end"] = max(
                    words[-1]["start"] + median_duration, segment["end"]
                )
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words
