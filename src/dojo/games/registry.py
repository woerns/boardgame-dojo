"""Game registry — discover and instantiate games by name."""

from __future__ import annotations

from typing import Any, Callable

from dojo.core.game import Game

_REGISTRY: dict[str, Callable[..., Game]] = {}


def register_game(name: str) -> Callable[[Callable[..., Game]], Callable[..., Game]]:
    """Decorator that registers a Game factory under *name*."""

    def decorator(factory: Callable[..., Game]) -> Callable[..., Game]:
        if name in _REGISTRY:
            raise ValueError(f"Game '{name}' is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def create_game(name: str, **kwargs: Any) -> Game:
    """Create a Game instance by its registered name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown game '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_games() -> list[str]:
    """Return sorted list of registered game names."""
    return sorted(_REGISTRY)
