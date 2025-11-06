"""Managers package initialization."""

from src.processing.managers.base import BaseManager
from src.processing.managers.cache import CacheManager
from src.processing.managers.file import FileManager
from src.processing.managers.processor import ProcessManager, ManagerHub
from src.processing.core.singleton import SingletonFactory

# Module-level singleton factories
_process_manager_factory: SingletonFactory[ProcessManager] = SingletonFactory(
    lambda: ProcessManager()
)


def get_process_manager(manager: ProcessManager | None = None) -> ProcessManager:
    """Get the default ProcessManager singleton, optionally providing an override."""
    return _process_manager_factory.get(manager)


__all__ = [
    "BaseManager",
    "CacheManager",
    "FileManager",
    "ProcessManager",
    "ManagerHub",
    "get_process_manager",
]
