import os
import subprocess
from langchain_core.messages import HumanMessage

from harness.react_agents.subagent_instances import (
    llm_code_interpreter_thinking_on,
    llm_doc_reader_thinking_on,
    llm_media_reader_thinking_on,
    llm_web_search_thinking_on,
)
from harness.utils.workspace import SHARED_DIR, reset_shared_directory


def run_test():
    print("=" * 70)
    print("GAIA SUBAGENT MODALITY TESTS (Single Subagent per Task)")
    print("=" * 70)

    # Ensure shared directory is initialized
    reset_shared_directory()

    # Generate a short local video in shared/ for the video test
    test_video_path = os.path.join(SHARED_DIR, "test_video.mp4")
    if not os.path.exists(test_video_path):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=4:size=640x360:rate=10",
                test_video_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    tests = [
        (
            "Task 1: SEE IMAGES (media_reader)",
            llm_media_reader_thinking_on,
            "Inspect the nutrition facts label in image 'data/gaia/b2c257e0-3ad7-4f05-b8e3-d9da973be36e.jpg'. "
            "What is the Serving Size and Calories per serving?",
        ),
        (
            "Task 2: HEAR AUDIO (media_reader)",
            llm_media_reader_thinking_on,
            "Transcribe the audio file 'data/gaia/1f975693-876d-457b-a649-393859e79bf3.mp3'. "
            "Which workbook pages are assigned for homework?",
        ),
        (
            "Task 3: WATCH VIDEO (media_reader)",
            llm_media_reader_thinking_on,
            f"Use analyze_video on the video file '{test_video_path}'. "
            "Describe the visual test pattern shown on screen and whether the rainbow bar at the bottom moves.",
        ),
        (
            "Task 4: VIEW PDF AS IMAGE (doc_reader)",
            llm_doc_reader_thinking_on,
            "Use render_document_to_image to render page 1 of the PDF document 'data/gaia/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf' "
            "into a high-resolution PNG image in shared/ for visual inspection. Report the path to the rendered image.",
        ),
        (
            "Task 5: USE PLAYWRIGHT BROWSER (web_search)",
            llm_web_search_thinking_on,
            "Navigate to 'https://example.com' using your browser navigation tool and extract the text from the webpage. "
            "What is the title and purpose of the domain mentioned on the page?",
        ),
        (
            "Task 6: SHELL TOOL / SANDBOX (code_interpreter)",
            llm_code_interpreter_thinking_on,
            "Use the bash shell tool to query Stockfish chess engine with the command 'stockfish uci | head -n 2'. "
            "Also compute and print 2**32 in python using python_interpreter. Output both results.",
        ),
    ]

    for title, subagent, prompt in [tests[5]]:
        print("\n" + "-" * 70)
        print(f"RUNNING: {title}")
        print(f"Prompt: {prompt}")
        print("-" * 70)
        try:
            response = subagent.react([HumanMessage(content=prompt)], task_id="test_suite")
            print(f"\nResult:\n{response}")
            print(f"\n✓ {title} COMPLETED SUCCESSFULLY")
        except Exception as e:
            print(f"\n✗ {title} FAILED: {e}")

    print("\n" + "=" * 70)
    print("ALL SINGLE SUBAGENT TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
