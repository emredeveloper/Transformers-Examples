"""Simple Gradio application for checking homework answers.

This version defaults to the light-weight ``distilgpt2`` model so the demo can
run on CPU-only environments without downloading multi-gigabyte weights. The
model name can be overridden either via a ``MODEL_NAME`` environment variable,
entries in a ``.env`` file or a ``--model-name`` CLI argument when launching the
app.

Heavy models still work, but the script prints additional guidance to help the
user understand that the download may take a long time and is best suited for a
GPU environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ENV_VARIABLE = "MODEL_NAME"
DEFAULT_MODEL_NAME = "distilgpt2"
DEFAULT_MAX_NEW_TOKENS = 256

_model = None
_tokenizer = None
_APP_SETTINGS = {"max_new_tokens": DEFAULT_MAX_NEW_TOKENS}


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Populate ``os.environ`` with variables defined in a local ``.env`` file."""

    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework checker demo app")
    parser.add_argument(
        "--model-name",
        help=(
            "Hugging Face model identifier to load. Overrides the MODEL_NAME "
            "environment variable if provided."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate for a single response.",
    )
    return parser.parse_args()


def _should_warn_about_heavy_model(model_name: str) -> bool:
    heavy_markers = ("phi", "mixtral", "llama", "mistral", "qwen", "deepseek")
    lowered = model_name.lower()
    return any(marker in lowered for marker in heavy_markers)


def _load_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if device.type == "cpu" and _should_warn_about_heavy_model(model_name):
        print(
            f"[INFO] `{model_name}` appears to be a large model. Running it on CPU "
            "may lead to very long download times and slow generation speeds."
        )
        print("[INFO] Consider using a GPU-backed environment for the best experience.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=device.type == "cpu",
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[INFO] Loaded model `{model_name}` on {device} with dtype={dtype}.")
    if _should_warn_about_heavy_model(model_name) and device.type == "cuda":
        print(
            "[INFO] This model may still take a while to download the first time, "
            "but the GPU will significantly speed up inference."
        )

    return model, tokenizer


# Function to validate the solution and provide feedback
def check_homework(exercise, solution, *, max_new_tokens: int) -> str:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model and tokenizer must be loaded before inference.")

    prompt = f"""
    Exercise: {exercise}
    Solution: {solution}
Task: Validate the solution to the math problem, provided by the user. If the user's solution is correct, confirm else provide an alternative if the solution is messy. If it is incorrect, provide the correct solution with step-by-step reasoning.
    """
    # Tokenize and generate response
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    print(f"Tokenized input length: {len(inputs['input_ids'][0])}")
    outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    print(f"Generated output length: {len(outputs[0])}")
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_len = len(prompt)
    response = response[prompt_len:].strip()
    print(f"Raw Response: {response}")
    return response


# Define the function that integrates with the Gradio app
def homework_checker_ui(exercise, solution):
    return check_homework(
        exercise,
        solution,
        max_new_tokens=_APP_SETTINGS["max_new_tokens"],
    )


def main() -> None:
    global _model, _tokenizer, _APP_SETTINGS

    _load_dotenv()
    args = _parse_args()

    model_name: str = args.model_name or os.environ.get(MODEL_ENV_VARIABLE, DEFAULT_MODEL_NAME)
    if args.model_name:
        print(f"[INFO] Using model name supplied via CLI argument: {model_name}")
    elif os.environ.get(MODEL_ENV_VARIABLE):
        print(f"[INFO] Using model name from environment: {model_name}")
    else:
        print(f"[INFO] Falling back to default model `{DEFAULT_MODEL_NAME}`")

    _model, _tokenizer = _load_model_and_tokenizer(model_name)

    max_new_tokens = args.max_new_tokens
    if max_new_tokens <= 0:
        print(
            f"[WARN] Received non-positive max_new_tokens={max_new_tokens}. "
            f"Resetting to default ({DEFAULT_MAX_NEW_TOKENS})."
        )
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    _APP_SETTINGS = {"max_new_tokens": max_new_tokens}

    interface = gr.Interface(
        fn=homework_checker_ui,
        inputs=[
            gr.Textbox(lines=2, label="Exercise (e.g., Solve for x in 2x + 3 = 7)"),
            gr.Textbox(lines=1, label="Your Solution (e.g., x = 1)"),
        ],
        outputs=gr.Textbox(label="Feedback"),
        title="AI Homework Checker",
        description="Validate your homework solutions, get corrections, and receive cleaner alternatives.",
    )

    interface.launch(debug=True)


if __name__ == "__main__":
    main()
