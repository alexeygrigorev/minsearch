"""
Field data containers for filtering.
"""
import pandas as pd
from dataclasses import dataclass


@dataclass
class FieldData:
    """Container for field names and their associated data."""

    fields: list[str]
    data: pd.DataFrame | dict[str, list]

    @property
    def num_docs(self) -> int:
        """Return the number of documents."""
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        else:
            return len(next(iter(self.data.values()))) if self.data else 0
