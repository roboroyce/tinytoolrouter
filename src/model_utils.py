"""Model/tokenizer helpers shared by the harness and the trainer.

Kept tiny and dependency-free at import time (transformers is imported by the
caller), so the stdlib-only test suite never needs it.
"""
from __future__ import annotations

import os


def is_local_path(model_id: str) -> bool:
    return os.path.isdir(model_id) or model_id.startswith((".", "/"))


def resolve_model_ref(model_id: str) -> str:
    """Turn a relative model dir into './...' so HF does not treat it as a repo id."""
    if os.path.isdir(model_id):
        return os.path.abspath(model_id)
    if "/" in model_id and not is_local_path(model_id) and " " not in model_id:
        # Looks like namespace/name -> genuine hub repo id.
        return model_id
    return model_id


def render_chat(tok, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """apply_chat_template that disables Qwen3-style *thinking* when supported.

    The router must answer directly; a thinking block wastes the token budget
    and makes the base model look artificially weak. We try the modern
    `enable_thinking=False` kwarg and fall back gracefully on tokenizers that
    do not accept it.
    """
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt, **kwargs)
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - template may reject the kwarg
            continue
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)