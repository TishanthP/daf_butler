# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import annotations

__all__ = ["CachingDimensionRecordStorage"]

from typing import Any, Dict, Iterable, Mapping, Optional, Union

import sqlalchemy

from lsst.utils import doImportType

from ...core import (
    DatabaseDimensionElement,
    DataCoordinateIterable,
    DimensionElement,
    DimensionRecord,
    GovernorDimension,
    HomogeneousDimensionRecordCache,
    NamedKeyDict,
    NamedKeyMapping,
    SpatialRegionDatabaseRepresentation,
    TimespanDatabaseRepresentation,
)
from ..interfaces import (
    Database,
    DatabaseDimensionRecordStorage,
    GovernorDimensionRecordStorage,
    StaticTablesContext,
)
from ..queries import QueryBuilder


class CachingDimensionRecordStorage(DatabaseDimensionRecordStorage):
    """A record storage implementation that adds caching to some other nested
    storage implementation.

    Parameters
    ----------
    nested : `DatabaseDimensionRecordStorage`
        The other storage to cache fetches from and to delegate all other
        operations to.
    """
    def __init__(self, nested: DatabaseDimensionRecordStorage):
        self._nested = nested
        self._cache = HomogeneousDimensionRecordCache(self.element, self._nested.fetch)

    @classmethod
    def initialize(
        cls,
        db: Database,
        element: DatabaseDimensionElement, *,
        context: Optional[StaticTablesContext] = None,
        config: Mapping[str, Any],
        governors: NamedKeyMapping[GovernorDimension, GovernorDimensionRecordStorage],
    ) -> DatabaseDimensionRecordStorage:
        # Docstring inherited from DatabaseDimensionRecordStorage.
        config = config["nested"]
        NestedClass = doImportType(config["cls"])
        if not hasattr(NestedClass, "initialize"):
            raise TypeError(f"Nested class {config['cls']} does not have an initialize() method.")
        nested = NestedClass.initialize(db, element, context=context, config=config, governors=governors)
        return cls(nested)

    @property
    def element(self) -> DatabaseDimensionElement:
        # Docstring inherited from DimensionRecordStorage.element.
        return self._nested.element

    def clearCaches(self) -> None:
        # Docstring inherited from DimensionRecordStorage.clearCaches.
        self._cache.clear()
        self._nested.clearCaches()

    def join(
        self,
        builder: QueryBuilder, *,
        regions: Optional[NamedKeyDict[DimensionElement, SpatialRegionDatabaseRepresentation]] = None,
        timespans: Optional[NamedKeyDict[DimensionElement, TimespanDatabaseRepresentation]] = None,
    ) -> None:
        # Docstring inherited from DimensionRecordStorage.
        return self._nested.join(builder, regions=regions, timespans=timespans)

    def insert(self, *records: DimensionRecord, replace: bool = False) -> None:
        # Docstring inherited from DimensionRecordStorage.insert.
        self._nested.insert(*records, replace=replace)
        self._cache.update(records)

    def sync(self, record: DimensionRecord, update: bool = False) -> Union[bool, Dict[str, Any]]:
        # Docstring inherited from DimensionRecordStorage.sync.
        inserted = self._nested.sync(record, update=update)
        if inserted:
            self._cache.add(record)
        return inserted

    def fetch(self, dataIds: DataCoordinateIterable) -> HomogeneousDimensionRecordCache:
        # Docstring inherited from DimensionRecordStorage.fetch.
        return self._cache.extract(dataIds)

    def digestTables(self) -> Iterable[sqlalchemy.schema.Table]:
        # Docstring inherited from DimensionRecordStorage.digestTables.
        return self._nested.digestTables()
