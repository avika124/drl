"""
Platform ad-network adapters.

Each adapter follows the same interface (``BasePlatformAdapter``) so the
DataPipeline can call any platform uniformly.
"""

from backend.adapters.base import BasePlatformAdapter  # noqa: F401
