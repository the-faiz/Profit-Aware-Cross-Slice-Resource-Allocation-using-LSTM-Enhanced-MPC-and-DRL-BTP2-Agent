from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs, deterministic: bool = True):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        raise NotImplementedError
