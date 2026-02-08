"""
Claude Code CLI client wrapper.

Replaces the Ollama client by shelling out to `claude -p` (non-interactive mode).
The Claude Code CLI must be installed and authenticated.
"""

import subprocess
import shutil
import json


def _find_claude_binary() -> str:
    """Locate the claude CLI binary."""
    path = shutil.which("claude")
    if path is None:
        raise FileNotFoundError(
            "claude CLI not found on PATH. "
            "Install it: https://docs.anthropic.com/en/docs/claude-code"
        )
    return path


def generate(prompt: str, system: str = None, model: str = None) -> str:
    """
    Send a prompt to Claude via the CLI and return the text response.

    Parameters
    ----------
    prompt : str
        The user prompt / input text.
    system : str, optional
        A system prompt that sets context and instructions.
    model : str, optional
        Model identifier to pass via --model flag (e.g. "sonnet", "opus").
        If None, the CLI uses its default model.

    Returns
    -------
    str
        The raw text response from Claude.
    """
    claude_bin = _find_claude_binary()

    cmd = [
        claude_bin,
        "-p",               # non-interactive prompt mode
        "--output-format", "text",
        "--max-turns", "1",  # single turn, no tool use
        "--no-input",        # don't read extra stdin
    ]

    if system:
        cmd.extend(["--system-prompt", system])

    if model:
        cmd.extend(["--model", model])

    # The prompt itself is the final positional argument
    cmd.append(prompt)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per call
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"claude CLI error (exit {result.returncode}): {stderr}")
            return ""

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        print("claude CLI timed out after 120s")
        return ""
    except Exception as e:
        print(f"Error calling claude CLI: {e}")
        return ""
