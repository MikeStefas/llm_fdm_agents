"""Media inspection tools: OCR, multimodal vision, and speech transcription."""

import base64
from io import BytesIO
import io
import os
import subprocess
import tempfile
import warnings

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import numpy as np
from PIL import Image
import pydub
from pytesseract import image_to_string
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from harness.llm import get_chat_model


def convert_to_base64(file_path: str) -> str:
    """Convert image to Base64 encoded JPEG string."""
    pil_image = Image.open(file_path)
    buffered = BytesIO()
    # Convert RGBA/palette to RGB if necessary for JPEG
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@tool
def ocr_image(
    path: str,
    prompt: str = "Transcribe all visible text, letters, numbers, and symbols in this image accurately and verbatim.",
    model_name: str | None = None,
) -> str:
    """Extracts and transcribes all text from an image using an OpenRouter multimodal vision LLM.

    Args:
        path: Path to the image file.
        prompt: Specific transcription instruction (defaults to verbatim text extraction).
        model_name: Optional vision model name.
    """
    if not os.path.exists(path):
        return f"Error: Image '{path}' not found."
    try:
        img_base64 = convert_to_base64(path)
        llm = get_chat_model(reasoning=False)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
        )
        response = llm.invoke([message])
        return str(response.content).strip() or "No text detected in image."
    except Exception as e:
        return f"Error extracting text with LLM: {e}"


@tool
def inspect_image(
    path: str,
    prompt: str = "Describe this image in detail. If there is text, charts, numbers, objects or diagrams, report them accurately.",
) -> str:
    """Inspects an image using an OpenRouter multimodal vision LLM.

    Args:
        path: Absolute or relative path to the image file.
        prompt: Specific question or instruction about what to look for in the image.
    """
    if not os.path.exists(path):
        return f"Error: Image '{path}' not found."
    try:
        img_base64 = convert_to_base64(path)
        llm = get_chat_model(reasoning=False)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
        )
        response = llm.invoke([message])
        return str(response.content)
    except Exception as e:
        return f"Error inspecting image '{path}': {e}"


@tool
def transcribe_audio(path: str) -> str:
    """Transcribes an audio file (.mp3, .wav, .m4a, .ogg) to text using HuggingFace Whisper model.

    Args:
        path: Path to the target audio file.
    """
    if not os.path.exists(path):
        return f"Error: Audio file '{path}' not found."
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

        # Load audio and resample to 16kHz mono float32
        audio = pydub.AudioSegment.from_file(path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        input_features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0].strip() or "No speech detected in audio file."
    except Exception as e:
        return f"Error transcribing audio '{path}': {e}"


MEDIA_READER_TOOLS = [
    ocr_image,
    inspect_image,
    transcribe_audio,
]