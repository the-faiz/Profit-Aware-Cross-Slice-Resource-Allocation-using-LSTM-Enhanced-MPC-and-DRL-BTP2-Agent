from __future__ import annotations

from typing import Any, Tuple


def require_gymnasium() -> Tuple[Any, Any]:
    try:
        import gymnasium as gym  # type: ignore
        from gymnasium import spaces  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "gymnasium is required. Install with: pip install gymnasium"
        ) from exc
    return gym, spaces


def require_torch() -> Any:
    try:
        import torch as th  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "torch is required. Install with: pip install torch"
        ) from exc
    return th
