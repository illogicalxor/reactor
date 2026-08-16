# Policy Signal

An English-only NLP service for US markets. It receives video or audio of a US
policy maker, transcribes the speech, and returns a `bullish` or `bearish` language
signal for a specified US asset. The response includes probabilities, signal
strength, the transcript, answer-level signals, and segment-level classifications.

`asset_context` is required conceptually because a statement can affect US equities,
Treasuries, and the US dollar differently. The default is `US equities`.

## Pipeline

```text
video/audio -> FFmpeg (16 kHz WAV) -> Whisper word timestamps -> speaker roles
            -> question/answer turns -> relevance gate -> ~30-second answers
            -> NLP classifier -> content-weighted recording aggregation
            -> answer series + bullish/bearish event aggregate + low_signal
```

Complete uploaded recordings use content weighting without temporal decay, so an
arbitrary closing remark cannot dominate the whole event. The rolling WebSocket
signal separately applies recency weighting over its approximately five-minute
window. Reporter speech and unknown acoustic speakers are excluded before stance
classification. Reporter questions are retained as `question_context` on the
following policy-maker answer, so the decision is auditable without counting the
question as the Chair's position.

The default `press-conference-text` role backend is a conservative fallback that
recognizes explicit reporter/outlet introductions. For actual acoustic speaker
separation, install and enable the optional pyannote backend:

```bash
pip install -e '.[diarization]'
```

Accept the conditions for `pyannote/speaker-diarization-community-1`, create a
Hugging Face access token, and configure:

```dotenv
POLICY_SIGNAL_SPEAKER_ROLE_BACKEND=pyannote
POLICY_SIGNAL_SPEAKER_DIARIZATION_TOKEN=hf_...
POLICY_SIGNAL_SPEAKER_DIARIZATION_DEVICE=cuda
```

The acoustic backend assigns anonymous speaker clusters and maps the speaker who
dominates the opening ten minutes to `policy_maker`. That assumption matches an FOMC
press conference, but it is role attribution, not biometric identity verification.
Set the backend to `none` for speeches already known to contain only the policy
maker.

The relevance gate classifies authority turns internally as `irrelevant`, `neutral`,
or `directional`; only directional turns reach the binary model. A heuristic remains
the portable default, and the repository also contains the separately trained
`models/policy-relevance-gate-v1` checkpoint. Enable it with:

```dotenv
POLICY_SIGNAL_RELEVANCE_BACKEND=fine-tuned
POLICY_SIGNAL_RELEVANCE_MODEL=models/policy-relevance-gate-v1
```

Set `POLICY_SIGNAL_RELEVANCE_BACKEND=none` only for controlled ablations.

Three classifier backends are available:

- `zero-shot` (default): English NLI checkpoint for cold-start use;
- `fine-tuned`: supervised checkpoint produced by the training command;
- `heuristic`: lightweight development and smoke-test baseline only.

## Run the service

Python 3.11–3.13 and FFmpeg are required. The Docker image includes both.

```bash
cp .env.example .env
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
uvicorn policy_signal.api:app --host 0.0.0.0 --port 8000
```

Or use Docker:

```bash
docker build -t policy-signal .
docker run --rm -p 8000:8000 --env-file .env policy-signal
```

Whisper and classifier checkpoints are downloaded on the first request. The default
CPU profile uses English `small.en` Whisper with `int8`. For NVIDIA/CUDA:

```dotenv
POLICY_SIGNAL_WHISPER_DEVICE=cuda
POLICY_SIGNAL_WHISPER_COMPUTE_TYPE=float16
```

The repository's offline bundle includes `models/whisper-small.en`. Its
`download_manifest.json` records the size and SHA-256 of every CTranslate2 file;
`model.bin` is stored through Git LFS. Running `policy-signal-fetch-whisper` again
refreshes both the checkpoint and this integrity manifest.

### Analyze a file

```bash
curl -X POST http://localhost:8000/v1/analyze-video \
  -F file=@fed-speech.mp4 \
  -F 'asset_context=S&P 500'
```

Abbreviated response:

```json
{
  "label": "bearish",
  "confidence": 0.78,
  "bullish_probability": 0.22,
  "bearish_probability": 0.78,
  "signal_strength": 0.56,
  "low_signal": false,
  "asset_context": "S&P 500",
  "transcript": "...",
  "language": "en",
  "duration_seconds": 94.2,
  "model": "faster-whisper:small.en + zero-shot:...",
  "directional_coverage": 0.61,
  "excluded_segment_count": 7,
  "directional_segment_count": 5,
  "policy_maker_segment_count": 9,
  "journalist_segment_count": 3,
  "unknown_speaker_segment_count": 0,
  "neutral_segment_count": 2,
  "irrelevant_segment_count": 2,
  "turns": [
    {
      "turn_id": "answer_0002",
      "question_turn_id": "question_0001",
      "label": "bearish",
      "bullish_probability": 0.18,
      "bearish_probability": 0.82,
      "low_signal": false,
      "directional_coverage": 0.73
    }
  ],
  "segments": []
}
```

`low_signal=true` means the API still returns one of the two contract labels, but
the probability difference is below the configured margin. Downstream consumers
should abstain from acting on those responses. Segment evidence includes
`speaker_id`, `speaker_role`, `role_confidence`, `role_source`, `turn_id`,
`question_turn_id`, `question_context`, `relevance_label`, and
`relevance_confidence`. The `turns` series summarizes every policy-maker statement
or answer within that analysis result. Neutral or irrelevant turns remain visible as
50/50, `low_signal=true` entries, while only directional evidence contributes to the
event aggregate.

## Real-time streaming

Connect to `ws://localhost:8000/v1/analyze-stream`. The first message is JSON:

```json
{
  "format": "pcm_s16le",
  "sample_rate": 16000,
  "window_seconds": 8,
  "language": "en",
  "asset_context": "S&P 500"
}
```

Then send binary frames containing mono signed 16-bit little-endian PCM extracted
from the video's audio track. The API processes 2–30 second windows and returns both
`window` (current window) and `rolling` (last five minutes) signals. Send
`{"command":"flush"}` to process buffered audio or `{"command":"end"}` to flush
and close. An `AudioWorklet` can extract PCM from a browser `<video>`/`MediaStream`;
FFmpeg or GStreamer can do this for a broadcast feed.

Each WebSocket connection owns isolated attribution state. Reporter questions may
span multiple audio windows, `turn_id` remains stable while the same answer
continues, and `question_turn_id` links the eventual answer to the complete
question. Acoustic pyannote cluster labels are still local to each diarization run;
use the text backend for stateful streaming until voiceprint-based cluster matching
is configured.

## Supervised training

Zero-shot inference is a starting point, not a trading-validated model. Install the
training dependencies and fine-tune on labeled US policy statements:

```bash
pip install -e '.[train]'
policy-signal-train \
  --train data/train.csv \
  --validation data/validation.csv \
  --output models/policy-signal-v1
```

The CSV must contain `text,label,asset_context`; labels must be `bullish` or
`bearish`. [data/example.csv](data/example.csv) documents the format but is far too
small for a useful model. The default English backbone is `FacebookAI/roberta-base`.

Serve the resulting checkpoint with:

```dotenv
POLICY_SIGNAL_CLASSIFIER_BACKEND=fine-tuned
POLICY_SIGNAL_CLASSIFIER_MODEL=models/policy-signal-v1
```

### Train the three-class relevance gate

Prepare chronological splits from the annotated Federal Reserve corpora and train
the gate separately from the binary direction model:

```bash
policy-signal-prepare-relevance --output data
policy-signal-train-relevance \
  --train data/relevance_train.csv \
  --validation data/relevance_validation.csv \
  --test data/relevance_test.csv \
  --dataset-manifest data/relevance_dataset_manifest.json \
  --base-model models/policy-signal-baseline-v2-large \
  --output models/policy-relevance-gate-v1
```

The trainer uses CUDA automatically when PyTorch detects it and records the device,
split hashes, selected checkpoint, and metrics in the output directory. The current
checkpoint reached test macro F1 `0.6734`, versus `0.3692` for the heuristic on the
same 59 examples. This test has only two `irrelevant` examples, so the per-class
result is not a stable production estimate.

### Offline training, inference-only production

Training and live inference are separate processes. The production image does not
install the optional training stack. On the training machine:

```bash
# Train and export model weights, tokenizer, config, and evaluation manifest.
policy-signal-train \
  --train data/train.csv \
  --validation data/validation.csv \
  --output artifacts/policy-signal-v1

# Fetch the ASR weights once; production will not download them.
policy-signal-fetch-whisper \
  --model small.en \
  --output artifacts/whisper-small.en

tar -czf policy-signal-v1.tar.gz -C artifacts \
  policy-signal-v1 whisper-small.en
sha256sum policy-signal-v1.tar.gz
```

Copy the archive to the inference host, verify its checksum, and extract it under a
versioned model directory. Do not copy only `model.safetensors`: the classifier
directory must also contain its tokenizer and model configuration.

Run production with read-only model mounts and local-only loading:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env.production \
  --mount type=bind,src=/srv/policy-models,dst=/models,readonly \
  --tmpfs /tmp \
  policy-signal
```

The settings in `.env.production.example` enable `LOCAL_MODELS_ONLY`, point both
components at `/models`, and prevent the service from treating a missing local path
as a remote model ID. They also enable `WARM_MODELS_ON_STARTUP`, so model loading is
paid during deployment rather than on the first live speech. Promote a new version by placing it in a new directory,
validating it, changing the configured path, and restarting the inference workers.
Never overwrite weights that a live process is currently using.

If a training job is interrupted, resume its optimizer and scheduler state:

```bash
policy-signal-train \
  --train data/train.csv \
  --validation data/validation.csv \
  --output artifacts/policy-signal-v1 \
  --resume-from-checkpoint artifacts/policy-signal-v1/checkpoint-1000
```

For the next periodic training cycle, keep production on `v1`, use the promoted
`v1` model as the next base model, and write to a new directory:

```bash
policy-signal-train \
  --base-model artifacts/policy-signal-v1 \
  --train data/train-v2.csv \
  --validation data/validation-v2.csv \
  --output artifacts/policy-signal-v2
```

Promote `v2` only after it beats `v1` on a fixed out-of-time test set and passes the
latency and calibration gates. The generated `training_manifest.json` records the
training parameters, evaluation metrics, and dataset SHA-256 hashes.

Split train/validation/test data by event and date, never by random statement. Keep
all excerpts from the same FOMC meeting or speech in one split. Define labels for a
specific US asset and horizon, and monitor per-class F1, calibration, latency, drift,
speaker performance, and out-of-time performance. Ambiguous observations should be
handled through `low_signal` because the external contract remains binary.

## Tests

Core tests do not download model checkpoints:

```bash
python -m unittest discover -s tests -v
```

This service measures directional language. It does not predict prices, model the
surprise relative to market consensus automatically, or provide financial advice.
