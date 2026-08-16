# Diarization validation

## Current environment

- Date: 16 August 2026
- GPU: NVIDIA Quadro T2000, 4 GB
- Python: 3.12.14
- pyannote.audio: 4.0.7
- Target checkpoint: `pyannote/speaker-diarization-community-1`

## Current result

The evaluation command and duration-based metrics are implemented and covered by
tests. Loading the real Community-1 checkpoint was attempted from this machine and
the Hugging Face Hub returned HTTP 401 because the repository is gated. No access
token is configured and no annotated real recording is present in the repository,
so reporting a real DER would be fabricated.

To complete the empirical run:

1. accept the Community-1 conditions on Hugging Face;
2. set `POLICY_SIGNAL_SPEAKER_DIARIZATION_TOKEN`;
3. provide the recording and a CSV reference timeline;
4. run `policy-signal-evaluate-diarization` as documented in the README.

The evaluator reports zero-collar exclusive diarization error rate, missed speech,
false alarm, speaker-role confusion, and duration-based precision/recall for
`policy_maker` and `journalist`. The opening-speaker anchor is role attribution, not
identity verification; voiceprint or visual confirmation remains necessary before
production promotion.
