from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Request:
    id: int
    bucket: str

@dataclass(frozen=True)
class Config:
    name: str
    tp: int
    capacity: int
