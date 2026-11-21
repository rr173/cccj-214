"""Catalog of declared relation schemas."""

from __future__ import annotations

from typing import Dict, Iterable

from .errors import SchemaError
from .types import Schema, schema_from_json


class SchemaCatalog:
    def __init__(self, schemas: Iterable[Schema]) -> None:
        self._schemas: Dict[str, Schema] = {}
        for s in schemas:
            if s.relation in self._schemas:
                raise SchemaError(f"duplicate schema for relation {s.relation!r}")
            self._schemas[s.relation] = s

    def get(self, relation: str) -> Schema:
        if relation not in self._schemas:
            raise SchemaError(f"unknown relation {relation!r}")
        return self._schemas[relation]

    @classmethod
    def from_json(cls, spec: dict) -> "SchemaCatalog":
        return cls(
            schema_from_json(rel, cols) for rel, cols in spec["relations"].items()
        )
