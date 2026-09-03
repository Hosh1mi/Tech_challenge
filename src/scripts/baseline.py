"""
Baseline evaluation script for GSM8K test set.
"""

import os

from pathlib import Path

def load_sys_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "system.prompt"
    with open(path, 'r') as f:
        return f.read().strip()

def load_usr_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "user.prompt"
    with open(path, 'r') as f:
        return f.read().strip()

