#!/usr/bin/env python3
"""Командний інтерфейс локальної української транскрипції."""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

from faster_whisper import WhisperModel


def timestamp(seconds: float) -> str:
    """Повертає компактний таймкод Г:ХХ:СС або Х:СС."""
    total_seconds = round(seconds)
    time = str(timedelta(seconds=total_seconds))
    hours, minutes, secs = time.split(":")
    return f"{int(hours)}:{minutes}:{secs}" if int(hours) else f"{int(minutes)}:{secs}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Транскрибувати відео українською з таймкодами."
    )
    parser.add_argument("video", type=Path, help="Шлях до відео або аудіофайлу")
    parser.add_argument(
        "-o", "--output", type=Path, help="Шлях до результату (типово: поруч із відео)"
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "plain"),
        default="markdown",
        help="Формат результату: markdown або plain для вставки в Notion (типово: markdown)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Модель Whisper: tiny, base, small, medium, large-v3 (типово: large-v3)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Пристрій"
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Точність обчислень, напр. int8 для CPU або float16 для CUDA",
    )
    return parser.parse_args()


def transcript_header(source: Path, language_probability: float, output_format: str) -> list[str]:
    """Повертає заголовок результату для обраного формату."""
    if output_format == "plain":
        return []
    return [
        f"# Транскрипція: {source.stem}",
        "",
        f"- Мова: українська ({language_probability:.0%} впевненість моделі)",
        f"- Джерело: `{source.name}`",
        "",
        "## Таймкоди",
        "",
    ]


def main() -> int:
    args = arguments()
    source = args.video.expanduser().resolve()
    if not source.is_file():
        print(f"Помилка: файл не знайдено: {source}", file=sys.stderr)
        return 2

    default_extension = ".md" if args.format == "markdown" else ".txt"
    output = args.output.expanduser().resolve() if args.output else source.with_suffix(default_extension)
    output.parent.mkdir(parents=True, exist_ok=True)

    # int8 добре підходить для локальної роботи на CPU та Apple Silicon.
    compute_type = args.compute_type or ("int8" if args.device in ("auto", "cpu") else "float16")
    print(f"Завантаження моделі {args.model}…", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    print("Розпізнавання української мови…", file=sys.stderr)
    segments, info = model.transcribe(
        str(source),
        language="uk",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    lines = transcript_header(source, info.language_probability, args.format)
    count = 0
    for segment in segments:
        text = segment.text.strip()
        if text:
            if args.format == "plain":
                lines.append(f"{timestamp(segment.start)} — {text}")
            else:
                lines.append(f"- **{timestamp(segment.start)}** — {text}")
            count += 1

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Готово: {output} ({count} фрагментів)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
