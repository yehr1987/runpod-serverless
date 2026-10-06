"""Patched handler for ivrit-ai/runpod-serverless (upstream infer.py + asr_options).

The stock worker calls faster-whisper with only language and word_timestamps; every other key in
transcribe_args is swallowed by **kwargs inside ivrit 0.2.6 and ignored. This version reads
transcribe_args["asr_options"] (initial_prompt, hotwords, beam_size, condition_on_previous_text,
vad_filter, temperature, ...) and makes them the defaults of the loaded model's transcribe().
Requests without asr_options behave exactly like the stock worker.
"""

import dataclasses
import functools
import queue
import sys
import threading
import runpod
import ivrit

_SENTINEL = object()
MAX_BATCH_SIZE = 20
ALLOWED_OPTIONS = {"initial_prompt", "hotwords", "beam_size", "best_of", "patience", "temperature",
                   "condition_on_previous_text", "vad_filter", "no_speech_threshold",
                   "compression_ratio_threshold", "log_prob_threshold", "repetition_penalty"}

current_model = None


def _with_defaults(orig, defaults, *args, **kwargs):
    merged = dict(defaults)
    merged.update(kwargs)          # what the library passes explicitly (language, word_timestamps) wins
    return orig(*args, **merged)


def _apply_asr_options(model, options):
    """Make options the default keyword arguments of the engine's own transcribe() call."""
    obj = getattr(model, "model_object", None)
    if obj is None:
        return
    if not hasattr(obj, "_orig_transcribe"):
        obj._orig_transcribe = obj.transcribe
    clean = {k: v for k, v in (options or {}).items() if k in ALLOWED_OPTIONS and v not in (None, "")}
    if clean:
        obj.transcribe = functools.partial(_with_defaults, obj._orig_transcribe, clean)
        print(f"asr_options applied: {sorted(clean)}", flush=True)
    else:
        obj.transcribe = obj._orig_transcribe


def transcribe(job):
    engine = job['input'].get('engine', 'faster-whisper')
    model_name = job['input'].get('model', None)
    is_streaming = job['input'].get('streaming', False)

    if engine not in ['faster-whisper', 'stable-whisper']:
        yield {"error": f"engine should be 'faster-whisper' or 'stable-whisper', but is {engine} instead."}
        return
    if not model_name:
        yield {"error": "Model not provided."}
        return
    transcribe_args = job['input'].get('transcribe_args', None)
    if not transcribe_args:
        yield {"error": "transcribe_args field not provided."}
        return
    if not ('blob' in transcribe_args or 'url' in transcribe_args):
        yield {"error": "transcribe_args must contain either 'blob' or 'url' field."}
        return

    stream_gen = transcribe_core(engine, model_name, transcribe_args)
    if is_streaming:
        for entry in stream_gen:
            yield entry
    else:
        yield {'result': [entry for entry in stream_gen]}


def transcribe_core(engine, model_name, transcribe_args):
    print('Transcribing...')
    global current_model

    different_model = (not current_model) or (current_model.engine != engine or current_model.model != model_name)
    if different_model:
        print(f'Loading new model: {engine} with {model_name}')
        current_model = ivrit.load_model(engine=engine, model=model_name, local_files_only=True)
    else:
        print(f'Reusing existing model: {engine} with {model_name}')

    _apply_asr_options(current_model, transcribe_args.pop('asr_options', None))   # must not reach ivrit's kwargs

    q = queue.Queue()

    def on_progress(event):
        q.put({"type": "progress", "data": event})

    transcribe_args['on_progress'] = on_progress
    diarize = transcribe_args.get('diarize', False)

    def producer():
        try:
            if diarize:
                res = current_model.transcribe(**transcribe_args)
                segs = res['segments']
            else:
                transcribe_args['stream'] = True
                segs = current_model.transcribe(**transcribe_args)
            for s in segs:
                q.put({"type": "segments", "data": [dataclasses.asdict(s)]})
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            batch = [item]
            while len(batch) < MAX_BATCH_SIZE and not q.empty():
                try:
                    more = q.get_nowait()
                    if more is _SENTINEL:
                        yield batch
                        return
                    if isinstance(more, Exception):
                        yield batch
                        raise more
                    batch.append(more)
                except queue.Empty:
                    break
            yield batch
    finally:
        thread.join()


import torch
if not torch.cuda.is_available():
    print("GPU health check failed: CUDA not available", flush=True)
    sys.exit(1)

runpod.serverless.start({"handler": transcribe, "return_aggregate_stream": True})
