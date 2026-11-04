import base64
from io import BytesIO
import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import numpy as np
from PIL import Image
import pydub
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from harness.utils.vcr_config import my_vcr


def convert_to_base64(file_path: str, max_dimension: int = 768) -> str:
    """Convert image to Base64 encoded JPEG string, downscaling if larger than max_dimension."""
    pil_image = Image.open(file_path)
    if max(pil_image.size) > max_dimension:
        pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    # Convert RGBA/palette to RGB if necessary for JPEG
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@tool
@my_vcr.use_cassette("ocr_image.yaml")
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
        img_base64 = convert_to_base64(path, max_dimension=768)
        llm = ChatOpenAI(
            model=model_name or "qwen/qwen3.5-9b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=1024,
            reasoning_effort="none",
            extra_body={
                "provider": {
                    "order": ["DeepInfra"],
                    "allow_fallbacks": False,
                }
            },
        )
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
@my_vcr.use_cassette("inspect_image.yaml")
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
        img_base64 = convert_to_base64(path, max_dimension=768)
        llm = ChatOpenAI(
            model="qwen/qwen3.5-9b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=1024,
            reasoning_effort="none",
            extra_body={
                "provider": {
                    "order": ["DeepInfra"],
                    "allow_fallbacks": False,
                }
            },
        )
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


_whisper_processor = None
_whisper_model = None


def _get_whisper():
    global _whisper_processor, _whisper_model
    if _whisper_processor is None or _whisper_model is None:
        _whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        _whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    return _whisper_processor, _whisper_model


@tool
def transcribe_audio(path: str) -> str:
    """Transcribes an audio file (.mp3, .wav, .m4a, .ogg) to text using HuggingFace Whisper model.

    Args:
        path: Path to the target audio file.
    """
    if not os.path.exists(path):
        return f"Error: Audio file '{path}' not found."
    try:
        processor, model = _get_whisper()

        # Load audio and resample to 16kHz mono float32
        audio = pydub.AudioSegment.from_file(path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        input_features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        text = transcription[0].strip()
        if not text:
            return "No speech detected in audio file."
        return f"Transcribed audio: {text}"
    except Exception as e:
        return f"Error transcribing audio '{path}': {e}"


MEDIA_READER_TOOLS = [
    ocr_image,
    inspect_image,
    transcribe_audio,
]