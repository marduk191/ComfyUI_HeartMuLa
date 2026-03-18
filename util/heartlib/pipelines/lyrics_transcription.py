import os

import torch
from huggingface_hub import snapshot_download
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from transformers.models.whisper.processing_whisper import WhisperProcessor
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)

_HF_REPO_TRANSCRIPTOR = "HeartMuLa/HeartTranscriptor-oss"


def _ensure_downloaded(local_dir: str, repo_id: str) -> None:
    """Download *repo_id* into *local_dir* if the directory is absent or empty."""
    if not os.path.isdir(local_dir) or not os.listdir(local_dir):
        print(f"[HeartMuLa] Downloading {repo_id} → {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local_dir)


class HeartTranscriptorPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "HeartTranscriptorPipeline":
        hearttranscriptor_path = os.path.join(pretrained_path, "HeartTranscriptor-oss")
        _ensure_downloaded(hearttranscriptor_path, _HF_REPO_TRANSCRIPTOR)

        model = WhisperForConditionalGeneration.from_pretrained(
            hearttranscriptor_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        processor = WhisperProcessor.from_pretrained(hearttranscriptor_path)

        return cls(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            dtype=dtype,
            chunk_length_s=30,
            batch_size=4,
        )
