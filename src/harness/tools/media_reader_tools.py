"""Media inspection tools: OCR, multimodal vision, and speech transcription."""

import base64
import io
import os
import subprocess
import tempfile
import warnings


from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import numpy as np
from PIL import Image
import pydub
from pytesseract import image_to_string
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def _load_image_data_url(path: str) -> str:
    """Loads an image with PIL and converts it to a standard base64 data URL."""
    with Image.open(path) as img:
        fmt = (img.format or "PNG").upper()
        save_fmt = "JPEG" if fmt in ("JPG", "JPEG") else ("PNG" if fmt in ("PNG", "WEBP") else "PNG")
        mime = "jpeg" if save_fmt == "JPEG" else "png"

        if img.mode not in ("RGB", "L") and save_fmt == "JPEG":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format=save_fmt)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{mime};base64,{encoded}"



@tool
def ocr_image(
    path: str,
    prompt: str = "Transcribe all visible text, letters, numbers, and symbols in this image accurately and verbatim.",
    model_name: str = "qwen3.5:2b",
) -> str:
    """Extracts and transcribes all text from an image using the local multimodal LLM (qwen3.5:2b).

    Args:
        path: Path to the image file.
        prompt: Specific transcription instruction (defaults to verbatim text extraction).
        model_name: Vision LLM model name in Ollama (defaults to qwen3.5:2b).
    """
    if not os.path.exists(path):
        return f"Error: Image '{path}' not found."
    try:
        data_url = _load_image_data_url(path)
        llm = ChatOllama(model=model_name, reasoning=False)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": data_url},
            ]
        )
        response = llm.invoke([message])
        return response.content.strip() or "No text detected in image."
    except Exception as e:
        return f"Error extracting text with LLM: {e}"


@tool
def inspect_image(
    path: str,
    prompt: str = "Describe this image in detail. If there is text, charts, numbers, objects or diagrams, report them accurately.",
    model_name: str = "qwen3.5:2b",
) -> str:
    """Inspects an image using the local multimodal LLM client (ChatOllama) combined with OCR.

    Args:
        path: Absolute or relative path to the image file.
        prompt: Specific question or instruction about what to look for in the image.
        model_name: Multimodal model name in Ollama (defaults to qwen3.5:2b).
    """
    if not os.path.exists(path):
        return f"Error: Image '{path}' not found."
    try:
        data_url = _load_image_data_url(path)
        llm = ChatOllama(model=model_name, reasoning=False)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": data_url},
            ]
        )
        response = llm.invoke([message])
        return response.content
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