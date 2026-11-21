import base64
import cv2
import glob
from io import BytesIO
import os
import shutil
import subprocess
import tempfile

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import numpy as np
from PIL import Image
import pydub
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from langchain_community.document_loaders import YoutubeLoader

from harness.utils.vcr_config import my_vcr
from harness.utils.workspace import SHARED_DIR
from config import PINNED_PROVIDER_EXTRA_BODY


def convert_to_base64(file_path: str, max_dimension: int | None = 2048) -> str:
    """Convert image to Base64 encoded JPEG string, preserving high resolution."""
    pil_image = Image.open(file_path)
    if max_dimension and max(pil_image.size) > max_dimension:
        pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    # Convert RGBA/palette to RGB if necessary for JPEG
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG", quality=90)
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
        img_base64 = convert_to_base64(path, max_dimension=2048)
        llm = ChatOpenAI(
            model=model_name or "qwen/qwen3.5-9b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=2048,
            request_timeout=90.0,
            max_retries=2,
            reasoning_effort="none",
            extra_body=PINNED_PROVIDER_EXTRA_BODY,
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
        img_base64 = convert_to_base64(path, max_dimension=2048)
        llm = ChatOpenAI(
            model="qwen/qwen3.5-9b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=2048,
            request_timeout=90.0,
            max_retries=2,
            reasoning_effort="none",
            extra_body=PINNED_PROVIDER_EXTRA_BODY,
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
    Audio longer than 30 seconds is automatically chunked into 30-second segments to capture the complete transcript.

    Args:
        path: Path to the target audio file.
    """
    clean_path = path.strip().strip("'\"")
    if not os.path.exists(clean_path):
        return f"Error: Audio file '{clean_path}' not found."
    try:
        processor, model = _get_whisper()

        # Load audio and resample to 16kHz mono float32
        audio = pydub.AudioSegment.from_file(clean_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        duration_ms = len(audio)
        if duration_ms == 0:
            return "No speech detected in audio file (empty audio)."

        chunk_ms = 30000  # 30-second segments
        transcriptions = []

        for start_ms in range(0, duration_ms, chunk_ms):
            chunk = audio[start_ms : start_ms + chunk_ms]
            samples = np.array(chunk.get_array_of_samples(), dtype=np.float32) / 32768.0
            if len(samples) == 0:
                continue

            input_features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features
            predicted_ids = model.generate(input_features)
            chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            if chunk_text:
                transcriptions.append(chunk_text)

        full_text = " ".join(transcriptions).strip()
        if not full_text:
            return "No speech detected in audio file."
        return f"Transcribed audio: {full_text}"
    except Exception as e:
        return f"Error transcribing audio '{clean_path}': {e}"


@tool
@my_vcr.use_cassette("youtube_transcript.yaml")
def get_youtube_transcript(url: str) -> str:
    """Fetches the official or auto-generated subtitle/transcript text of a YouTube video URL using LangChain's YoutubeLoader.

    Args:
        url: Full YouTube video URL (e.g. 'https://www.youtube.com/watch?v=...').
    """
    try:

        loader = YoutubeLoader.from_youtube_url(
            url.strip(),
            add_video_info=False,
            language=["en", "en-US", "en-GB", "el", "auto"],
        )
        docs = loader.load()
        if not docs or not docs[0].page_content.strip():
            return "No transcript or captions available for this YouTube video."
        return f"YouTube Transcript: {docs[0].page_content[:4000]}"
    except Exception as e:
        return f"Could not retrieve YouTube transcript for '{url}': {e}"


@tool
@my_vcr.use_cassette("analyze_video.yaml")
def analyze_video(
    video_url: str,
    prompt: str = "These are frames from a video. Create a short description of the video content, answering any specific question asked.",
) -> str:
    """Visually analyzes a video (YouTube URL or local video file) using OpenCV (cv2) frame extraction
    and our multimodal vision LLM following the OpenAI Cookbook standard.

    Args:
        video_url: Local video file path or YouTube/web video URL.
        prompt: Specific question or instruction about the video content.
    """
    raw_target = video_url.strip().strip("'\"")
    local_path = raw_target
    if local_path.startswith("file://"):
        local_path = local_path[7:]

    temp_dl_dir = None
    try:
        # If it's a web / YouTube URL, download via yt-dlp first
        if not os.path.exists(local_path):
            if raw_target.startswith("http://") or raw_target.startswith("https://"):
                temp_dl_dir = tempfile.mkdtemp(dir=SHARED_DIR, prefix="ytdl_")
                dl_out = os.path.join(temp_dl_dir, "downloaded.%(ext)s")
                cmd_dl = [
                    "yt-dlp",
                    "-f", "best[height<=480]/best[ext=mp4]/best",
                    "-o", dl_out,
                    "--no-playlist",
                    "--max-filesize", "50M",
                    raw_target,
                ]
                res_dl = subprocess.run(cmd_dl, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
                downloaded_files = glob.glob(os.path.join(temp_dl_dir, "downloaded.*"))
                if downloaded_files:
                    local_path = downloaded_files[0]
                else:
                    return f"Error downloading video from '{video_url}' with yt-dlp: {res_dl.stderr}"
            else:
                return f"Error: Video file '{video_url}' not found."

        # Extract base64 frames using OpenCV (cv2)
        video = cv2.VideoCapture(local_path)
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        video.release()

        if not base64Frames:
            return f"Error: Could not read video frames from '{video_url}'."

        #sample every 50th frame
        step = 50 if len(base64Frames) >= 50 else max(1, len(base64Frames) // 5)
        selected_frames = base64Frames[0::step]

        # map frames into image_url content list
        content = [
            {"type": "text", "text": prompt},
            *map(
                lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}},
                selected_frames,
            ),
        ]

        llm = ChatOpenAI(
            model="qwen/qwen3.5-9b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.0,
            max_tokens=2048,
            request_timeout=90.0,
            max_retries=2,
            reasoning_effort="none",
            extra_body=PINNED_PROVIDER_EXTRA_BODY,
        )
        response = llm.invoke([HumanMessage(content=content)])
        return str(response.content).strip() or "Video analysis produced no visual description."
    except Exception as e:
        return f"Error analyzing video '{video_url}': {e}"
    finally:
        if temp_dl_dir and os.path.exists(temp_dl_dir):
            shutil.rmtree(temp_dl_dir, ignore_errors=True)


MEDIA_READER_TOOLS = [
    ocr_image,
    inspect_image,
    transcribe_audio,
    get_youtube_transcript,
    analyze_video,
]