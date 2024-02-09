from dataclasses import dataclass
from typing import Optional, Self


@dataclass(repr=False)
class Node:
    parent: Optional[Self]
    tasks: tuple[int, ...]
