from __future__ import annotations

__all__ = ["StaticTablesTuple", "CollectionType", "DatasetUniqueness", "STATIC_TABLES_SPEC"]

import enum
import hashlib
from namedtuple import namedtuple

import sqlalchemy

from ..core.dimensions import DimensionGraph
from ..core.dimensions.schema import TIMESPAN_FIELD_SPECS, addDimensionForeignKey
from ..core.schema import TableSpec, FieldSpec, ForeignKeySpec, Base64Bytes


StaticTablesTuple = namedtuple(
    "StaticTablesTuple",
    [
        "collection",
        "dataset_composition",
        "dataset_location",
        "dataset_type_dimension",
        "dataset_type",
        "dataset",
        "layer_meta_dimension",
        "run",
    ]
)


class CollectionType(enum.IntEnum):
    RUN = 1
    TAGGED = 2
    CALIBRATION = 3


class DatasetUniqueness(enum.IntEnum):
    STANDARD = 1
    NONSINGULAR = 2
    GLOBAL = 3


STATIC_TABLES_SPEC = StaticTablesTuple(
    collection=TableSpec(
        fields=[
            FieldSpec("id", dtype=sqlalchemy.BigInteger, autoincrement=True, primaryKey=True),
            FieldSpec("origin", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("name", dtype=sqlalchemy.String, length=64, nullable=False),
            FieldSpec("type", dtype=sqlalchemy.SmallInteger, nullable=False),
        ],
        unique={("name",)},
    ),
    dataset_composition=TableSpec(
        fields=[
            FieldSpec("parent_dataset_id", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("parent_origin", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("component_dataset_id", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("component_origin", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("component_name", dtype=sqlalchemy.String, length=32),
        ],
        foreignKeys=[
            ForeignKeySpec("dataset", source=("parent_dataset_id", "parent_origin"),
                           target=("id", "origin"), onDelete="CASCADE"),
            ForeignKeySpec("dataset", source=("component_dataset_id", "component_origin"),
                           target=("id", "origin"), onDelete="CASCADE"),
        ]
    ),
    dataset_location=TableSpec(
        fields=[
            FieldSpec("dataset_id", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("origin", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("datastore_name", dtype=sqlalchemy.String, length=256, primaryKey=True),
        ],
        foreignKeys=[
            ForeignKeySpec("dataset", source=("dataset_id", "origin"), target=("id", "origin"),
                           onDelete="CASCADE"),
        ]
    ),
    dataset_type_dimension=TableSpec(
        fields=[
            FieldSpec("dataset_type_name", dtype=sqlalchemy.String, length=128, primaryKey=True),
            FieldSpec("dimension_name", dtype=sqlalchemy.String, length=32, primaryKey=True),
        ],
        foreignKeys=[
            ForeignKeySpec("dataset_type", source=("dataset_type_name",), target=("name"),
                           onDelete="CASCADE"),
        ]
    ),
    dataset_type=TableSpec(
        fields=[
            FieldSpec("name", dtype=sqlalchemy.String, length=128, primaryKey=True),
            FieldSpec("storage_class", dtype=sqlalchemy.String, length=64, nullable=False),
            FieldSpec("uniqueness", dtype=sqlalchemy.SmallInteger, nullable=False),
        ],
    ),
    dataset=TableSpec(
        fields=[
            FieldSpec("id", dtype=sqlalchemy.BigInteger, autoincrement=True, primaryKey=True),
            FieldSpec("origin", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("dataset_type_name", dtype=sqlalchemy.String, length=128),
            FieldSpec("dataset_ref_hash", dtype=Base64Bytes, nbytes=32),
            FieldSpec("run_collection_id", dtype=sqlalchemy.BigInteger, nullable=False),
            FieldSpec("quantum_id", dtype=sqlalchemy.BigInteger),
        ],
        foreignKeys=[
            ForeignKeySpec("dataset_type", source=("dataset_type_name",), target=("name")),
            ForeignKeySpec("run", source=("run_collection_id", "origin"), target=("collection_id", "origin"),
                           onDelete="CASCADE"),
        ]
    ),
    layer_meta_dimension=TableSpec(
        fields=[
            FieldSpec("element_name", dtype=sqlalchemy.String, length=64, primaryKey=True),
        ],
    ),
    run=TableSpec(
        fields=[
            FieldSpec("collection_id", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("origin", dtype=sqlalchemy.BigInteger, primaryKey=True),
            FieldSpec("name", dtype=sqlalchemy.String, length=64, nullable=False),
            TIMESPAN_FIELD_SPECS.begin,
            TIMESPAN_FIELD_SPECS.end,
            FieldSpec("host", dtype=sqlalchemy.String, length=128),
        ],
        unique={("name",)},
        foreignKeys=[
            ForeignKeySpec("collection", source=("collection_id", "origin"), target=("id", "origin"),
                           onDelete="CASCADE"),
        ],
    ),
)
