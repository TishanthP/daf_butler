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

__all__ = ("DatasetRecordStorageManager", "DatasetRecordStorage", "DatasetIdGenEnum")

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AbstractSet, Any, Iterable, Iterator, Optional, Tuple

from ...core import DataCoordinate, DatasetId, DatasetRef, DatasetType, Timespan, ddl, sql
from ...core.named import NamedValueAbstractSet
from ._versioning import VersionedExtension

if TYPE_CHECKING:
    from .._collection_summary import CollectionSummary
    from ._collections import CollectionManager, CollectionRecord, RunRecord
    from ._database import Database, StaticTablesContext
    from ._dimensions import DimensionRecordStorageManager


class DatasetIdGenEnum(enum.Enum):
    """This enum is used to specify dataset ID generation options for
    ``insert()`` method.
    """

    UNIQUE = 0
    """Unique mode generates unique ID for each inserted dataset, e.g.
    auto-generated by database or random UUID.
    """

    DATAID_TYPE = 1
    """In this mode ID is computed deterministically from a combination of
    dataset type and dataId.
    """

    DATAID_TYPE_RUN = 2
    """In this mode ID is computed deterministically from a combination of
    dataset type, dataId, and run collection name.
    """


class DatasetRecordStorage(ABC):
    """An interface that manages the records associated with a particular
    `DatasetType`.

    Parameters
    ----------
    datasetType : `DatasetType`
        Dataset type whose records this object manages.
    """

    def __init__(self, datasetType: DatasetType):
        self.datasetType = datasetType

    @abstractmethod
    def insert(
        self,
        run: RunRecord,
        dataIds: Iterable[DataCoordinate],
        idGenerationMode: DatasetIdGenEnum = DatasetIdGenEnum.UNIQUE,
    ) -> Iterator[DatasetRef]:
        """Insert one or more dataset entries into the database.

        Parameters
        ----------
        run : `RunRecord`
            The record object describing the `~CollectionType.RUN` collection
            this dataset will be associated with.
        dataIds : `Iterable` [ `DataCoordinate` ]
            Expanded data IDs (`DataCoordinate` instances) for the
            datasets to be added.   The dimensions of all data IDs must be the
            same as ``self.datasetType.dimensions``.
        idMode : `DatasetIdGenEnum`
            With `UNIQUE` each new dataset is inserted with its new unique ID.
            With non-`UNIQUE` mode ID is computed from some combination of
            dataset type, dataId, and run collection name; if the same ID is
            already in the database then new record is not inserted.

        Returns
        -------
        datasets : `Iterable` [ `DatasetRef` ]
            References to the inserted datasets.
        """
        raise NotImplementedError()

    @abstractmethod
    def import_(
        self,
        run: RunRecord,
        datasets: Iterable[DatasetRef],
        idGenerationMode: DatasetIdGenEnum = DatasetIdGenEnum.UNIQUE,
        reuseIds: bool = False,
    ) -> Iterator[DatasetRef]:
        """Insert one or more dataset entries into the database.

        Parameters
        ----------
        run : `RunRecord`
            The record object describing the `~CollectionType.RUN` collection
            this dataset will be associated with.
        datasets :  `~collections.abc.Iterable` of `DatasetRef`
            Datasets to be inserted.  Datasets can specify ``id`` attribute
            which will be used for inserted datasets. All dataset IDs must
            have the same type (`int` or `uuid.UUID`), if type of dataset IDs
            does not match type supported by this class then IDs will be
            ignored and new IDs will be generated by backend.
        idGenerationMode : `DatasetIdGenEnum`
            With `UNIQUE` each new dataset is inserted with its new unique ID.
            With non-`UNIQUE` mode ID is computed from some combination of
            dataset type, dataId, and run collection name; if the same ID is
            already in the database then new record is not inserted.
        reuseIds : `bool`, optional
            If `True` then forces re-use of imported dataset IDs for integer
            IDs which are normally generated as auto-incremented; exception
            will be raised if imported IDs clash with existing ones. This
            option has no effect on the use of globally-unique IDs which are
            always re-used (or generated if integer IDs are being imported).

        Returns
        -------
        datasets : `Iterable` [ `DatasetRef` ]
            References to the inserted or existing datasets.

        Notes
        -----
        The ``datasetType`` and ``run`` attributes of datasets are supposed to
        be identical across all datasets but this is not checked and it should
        be enforced by higher level registry code. This method does not need
        to use those attributes from datasets, only ``dataId`` and ``id`` are
        relevant.
        """
        raise NotImplementedError()

    @abstractmethod
    def find(
        self, collection: CollectionRecord, dataId: DataCoordinate, timespan: Optional[Timespan] = None
    ) -> Optional[DatasetRef]:
        """Search a collection for a dataset with the given data ID.

        Parameters
        ----------
        collection : `CollectionRecord`
            The record object describing the collection to search for the
            dataset.  May have any `CollectionType`.
        dataId: `DataCoordinate`
            Complete (but not necessarily expanded) data ID to search with,
            with ``dataId.graph == self.datasetType.dimensions``.
        timespan : `Timespan`, optional
            A timespan that the validity range of the dataset must overlap.
            Required if ``collection.type is CollectionType.CALIBRATION``, and
            ignored otherwise.

        Returns
        -------
        ref : `DatasetRef`
            A resolved `DatasetRef` (without components populated), or `None`
            if no matching dataset was found.
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, datasets: Iterable[DatasetRef]) -> None:
        """Fully delete the given datasets from the registry.

        Parameters
        ----------
         datasets : `Iterable` [ `DatasetRef` ]
            Datasets to be deleted.  All datasets must be resolved and have
            the same `DatasetType` as ``self``.

        Raises
        ------
        AmbiguousDatasetError
            Raised if any of the given `DatasetRef` instances is unresolved.
        """
        raise NotImplementedError()

    @abstractmethod
    def associate(self, collection: CollectionRecord, datasets: Iterable[DatasetRef]) -> None:
        """Associate one or more datasets with a collection.

        Parameters
        ----------
        collection : `CollectionRecord`
            The record object describing the collection.  ``collection.type``
            must be `~CollectionType.TAGGED`.
        datasets : `Iterable` [ `DatasetRef` ]
            Datasets to be associated.  All datasets must be resolved and have
            the same `DatasetType` as ``self``.

        Raises
        ------
        AmbiguousDatasetError
            Raised if any of the given `DatasetRef` instances is unresolved.

        Notes
        -----
        Associating a dataset with into collection that already contains a
        different dataset with the same `DatasetType` and data ID will remove
        the existing dataset from that collection.

        Associating the same dataset into a collection multiple times is a
        no-op, but is still not permitted on read-only databases.
        """
        raise NotImplementedError()

    @abstractmethod
    def disassociate(self, collection: CollectionRecord, datasets: Iterable[DatasetRef]) -> None:
        """Remove one or more datasets from a collection.

        Parameters
        ----------
        collection : `CollectionRecord`
            The record object describing the collection.  ``collection.type``
            must be `~CollectionType.TAGGED`.
        datasets : `Iterable` [ `DatasetRef` ]
            Datasets to be disassociated.  All datasets must be resolved and
            have the same `DatasetType` as ``self``.

        Raises
        ------
        AmbiguousDatasetError
            Raised if any of the given `DatasetRef` instances is unresolved.
        """
        raise NotImplementedError()

    @abstractmethod
    def certify(
        self, collection: CollectionRecord, datasets: Iterable[DatasetRef], timespan: Timespan
    ) -> None:
        """Associate one or more datasets with a calibration collection and a
        validity range within it.

        Parameters
        ----------
        collection : `CollectionRecord`
            The record object describing the collection.  ``collection.type``
            must be `~CollectionType.CALIBRATION`.
        datasets : `Iterable` [ `DatasetRef` ]
            Datasets to be associated.  All datasets must be resolved and have
            the same `DatasetType` as ``self``.
        timespan : `Timespan`
            The validity range for these datasets within the collection.

        Raises
        ------
        AmbiguousDatasetError
            Raised if any of the given `DatasetRef` instances is unresolved.
        ConflictingDefinitionError
            Raised if the collection already contains a different dataset with
            the same `DatasetType` and data ID and an overlapping validity
            range.
        CollectionTypeError
            Raised if
            ``collection.type is not CollectionType.CALIBRATION`` or if
            ``self.datasetType.isCalibration() is False``.
        """
        raise NotImplementedError()

    @abstractmethod
    def decertify(
        self,
        collection: CollectionRecord,
        timespan: Timespan,
        *,
        dataIds: Optional[Iterable[DataCoordinate]] = None,
    ) -> None:
        """Remove or adjust datasets to clear a validity range within a
        calibration collection.

        Parameters
        ----------
        collection : `CollectionRecord`
            The record object describing the collection.  ``collection.type``
            must be `~CollectionType.CALIBRATION`.
        timespan : `Timespan`
            The validity range to remove datasets from within the collection.
            Datasets that overlap this range but are not contained by it will
            have their validity ranges adjusted to not overlap it, which may
            split a single dataset validity range into two.
        dataIds : `Iterable` [ `DataCoordinate` ], optional
            Data IDs that should be decertified within the given validity range
            If `None`, all data IDs for ``self.datasetType`` will be
            decertified.

        Raises
        ------
        CollectionTypeError
            Raised if ``collection.type is not CollectionType.CALIBRATION``.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_relation(
        self,
        *collections: CollectionRecord,
        columns: AbstractSet[str],
        constraints: Optional[sql.LocalConstraints] = None,
    ) -> sql.Relation:
        """Return a `sql.Relation` that represents a query for for this
        `DatasetType` in one or more collections.

        Parameters
        ----------
        *collections : `CollectionRecord`
            The record object(s) describing the collection(s) to query.  May
            not be of type `CollectionType.CHAINED`.  If multiple collections
            are passed, the query will search all of them in an unspecified
            order, and all collections must have the same type.  Must include
            at least one collection.
        columns : `AbstractSet` [ `str` ]
            Columns to include in the relation.  See
            `QueryBackend.make_dataset_query_relation` for most options, but
            this method supports one more:

            - ``rank``: a calculated integer column holding the index of the
              collection the dataset was found in, within the ``collections``
              sequence given.

        constraints : `LocalConstraints`, optional
            Query-time constraints on the query result rows that are known
            in advance.  Passing `None` (default) is equivalent to passing
            a call to `sql.LocalConstraints.make_full`.

        Returns
        ------
        relation : `sql.Relation`
            Representation of the query.
        """
        raise NotImplementedError()

    datasetType: DatasetType
    """Dataset type whose records this object manages (`DatasetType`).
    """


class DatasetRecordStorageManager(VersionedExtension):
    """An interface that manages the tables that describe datasets.

    `DatasetRecordStorageManager` primarily serves as a container and factory
    for `DatasetRecordStorage` instances, which each provide access to the
    records for a different `DatasetType`.
    """

    @classmethod
    @abstractmethod
    def initialize(
        cls,
        db: Database,
        context: StaticTablesContext,
        *,
        collections: CollectionManager,
        dimensions: DimensionRecordStorageManager,
        column_types: sql.ColumnTypeInfo,
    ) -> DatasetRecordStorageManager:
        """Construct an instance of the manager.

        Parameters
        ----------
        db : `Database`
            Interface to the underlying database engine and namespace.
        context : `StaticTablesContext`
            Context object obtained from `Database.declareStaticTables`; used
            to declare any tables that should always be present.
        collections: `CollectionManager`
            Manager object for the collections in this `Registry`.
        dimensions : `DimensionRecordStorageManager`
            Manager object for the dimensions in this `Registry`.
        column_types : `sql.ColumnTypeInfo`
            Information about column types that can differ between data
            repositories and registry instances.

        Returns
        -------
        manager : `DatasetRecordStorageManager`
            An instance of a concrete `DatasetRecordStorageManager` subclass.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def getIdColumnType(cls) -> type:
        """Return type used for columns storing dataset IDs.

        This type is used for columns storing `DatasetRef.id` values, usually
        a `type` subclass provided by SQLAlchemy.

        Returns
        -------
        dtype : `type`
            Type used for dataset identification in database.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def supportsIdGenerationMode(cls, mode: DatasetIdGenEnum) -> bool:
        """Test whether the given dataset ID generation mode is supported by
        `insert`.

        Parameters
        ----------
        mode : `DatasetIdGenEnum`
            Enum value for the mode to test.

        Returns
        -------
        supported : `bool`
            Whether the given mode is supported.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def addDatasetForeignKey(
        cls,
        tableSpec: ddl.TableSpec,
        *,
        prefix: str = "dataset",
        constraint: bool = True,
        onDelete: Optional[str] = None,
        exact_name: Optional[str] = None,
        **kwargs: Any,
    ) -> ddl.FieldSpec:
        """Add a foreign key (field and constraint) referencing the dataset
        table.

        Parameters
        ----------
        tableSpec : `ddl.TableSpec`
            Specification for the table that should reference the dataset
            table.  Will be modified in place.
        prefix: `str`, optional
            A name to use for the prefix of the new field; the full name is
            ``{prefix}_id``.
        onDelete: `str`, optional
            One of "CASCADE" or "SET NULL", indicating what should happen to
            the referencing row if the collection row is deleted.  `None`
            indicates that this should be an integrity error.
        constraint: `bool`, optional
            If `False` (`True` is default), add a field that can be joined to
            the dataset primary key, but do not add a foreign key constraint.
        exact_name : `str`, optional
            Complete name for the field, overriding ``prefix``.
        **kwargs
            Additional keyword arguments are forwarded to the `ddl.FieldSpec`
            constructor (only the ``name`` and ``dtype`` arguments are
            otherwise provided).

        Returns
        -------
        idSpec : `ddl.FieldSpec`
            Specification for the ID field.
        """
        raise NotImplementedError()

    @abstractmethod
    def refresh(self) -> None:
        """Ensure all other operations on this manager are aware of any
        dataset types that may have been registered by other clients since
        it was initialized or last refreshed.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def parent_dataset_types(self) -> NamedValueAbstractSet[DatasetType]:
        """A set view of all parent dataset types known to the manager
        (`NamedValueAbstractSet`).
        """
        raise NotImplementedError()

    def __getitem__(self, name: str) -> DatasetRecordStorage:
        """Return the object that provides access to the records associated
        with the given `DatasetType` name.

        This is simply a convenience wrapper for `find` that raises `KeyError`
        when the dataset type is not found.

        Returns
        -------
        records : `DatasetRecordStorage`
            The object representing the records for the given dataset type.

        Raises
        ------
        KeyError
            Raised if there is no dataset type with the given name.

        Notes
        -----
        Dataset types registered by another client of the same repository since
        the last call to `initialize` or `refresh` may not be found.
        """
        result = self.find(name)
        if result is None:
            raise KeyError(f"Dataset type with name '{name}' not found.")
        return result

    @abstractmethod
    def find(self, name: str) -> Optional[DatasetRecordStorage]:
        """Return an object that provides access to the records associated with
        the given `DatasetType` name, if one exists.

        Parameters
        ----------
        name : `str`
            Name of the dataset type.

        Returns
        -------
        records : `DatasetRecordStorage` or `None`
            The object representing the records for the given dataset type, or
            `None` if there are no records for that dataset type.

        Notes
        -----
        Dataset types registered by another client of the same repository since
        the last call to `initialize` or `refresh` may not be found.
        """
        raise NotImplementedError()

    @abstractmethod
    def register(self, datasetType: DatasetType) -> Tuple[DatasetRecordStorage, bool]:
        """Ensure that this `Registry` can hold records for the given
        `DatasetType`, creating new tables as necessary.

        Parameters
        ----------
        datasetType : `DatasetType`
            Dataset type for which a table should created (as necessary) and
            an associated `DatasetRecordStorage` returned.

        Returns
        -------
        records : `DatasetRecordStorage`
            The object representing the records for the given dataset type.
        inserted : `bool`
            `True` if the dataset type did not exist in the registry before.

        Notes
        -----
        This operation may not be invoked within a `Database.transaction`
        context.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, name: str) -> None:
        """Remove the dataset type.

        Parameters
        ----------
        name : `str`
            Name of the dataset type.
        """
        raise NotImplementedError()

    def __iter__(self) -> Iterator[DatasetType]:
        """Return an iterator over the the dataset types present in this layer.

        Notes
        -----
        Dataset types registered by another client of the same layer since
        the last call to `initialize` or `refresh` may not be included.
        """
        return iter(self.parent_dataset_types)

    @abstractmethod
    def getDatasetRef(self, id: DatasetId) -> Optional[DatasetRef]:
        """Return a `DatasetRef` for the given dataset primary key
        value.

        Parameters
        ----------
        id : `DatasetId`
            Primary key value for the dataset.

        Returns
        -------
        ref : `DatasetRef` or `None`
            Object representing the dataset, or `None` if no dataset with the
            given primary key values exists in this layer.
        """
        raise NotImplementedError()

    @abstractmethod
    def getCollectionSummary(self, collection: CollectionRecord) -> CollectionSummary:
        """Return a summary for the given collection.

        Parameters
        ----------
        collection : `CollectionRecord`
            Record describing the collection for which a summary is to be
            retrieved.

        Returns
        -------
        summary : `CollectionSummary`
            Summary of the dataset types and governor dimension values in
            this collection.
        """
        raise NotImplementedError()
