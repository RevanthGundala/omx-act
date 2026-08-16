from __future__ import annotations

import os
import sys
from typing import TextIO

_STATUS_COLORS = {
    "green": "\033[1;32m",
    "yellow": "\033[1;33m",
    "red": "\033[1;31m",
}
_RESET = "\033[0m"


def colorize_status(text: str, color: str, output: TextIO | None = None) -> str:
    output = sys.stdout if output is None else output
    if os.environ.get("NO_COLOR") is not None or not output.isatty():
        return text
    return f"{_STATUS_COLORS[color]}{text}{_RESET}"
