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

__all__ = ["DatasetRef"]

from copy import deepcopy
import hashlib

from types import MappingProxyType
from ..utils import slotValuesAreEqual
from ..dimensions import DataCoordinate
from .type import DatasetType


def _safeMakeMappingProxyType(data):
    if data is None:
        data = {}
    return MappingProxyType(data)


class DatasetRef:
    """Reference to a Dataset in a `Registry`.

    A `DatasetRef` may point to a Dataset that currently does not yet exist
    (e.g., because it is a predicted input for provenance).

    Parameters
    ----------
    datasetType : `DatasetType`
        The `DatasetType` for this Dataset.
    dataId : `DataCoordinate`
        A mapping of dimensions that labels the Dataset within a Collection.
    id : `int`, optional
        A unique identifier.
        Normally set to `None` and assigned by `Registry`
    """

    __slots__ = ("_id", "_datasetType", "_dataId", "_producer", "_run", "_hash",
                 "_predictedConsumers", "_actualConsumers", "_components")

    def __init__(self, datasetType, dataId, *, id=None, run=None, hash=None, components=None, conform=True):
        assert isinstance(datasetType, DatasetType)
        self._id = id
        self._datasetType = datasetType
        if conform:
            self._dataId = DataCoordinate.standardize(dataId, graph=datasetType.dimensions)
        else:
            self._dataId = dataId
        self._producer = None
        self._predictedConsumers = dict()
        self._actualConsumers = dict()
        self._components = dict()
        if components is not None:
            self._components.update(components)
        self._run = run
        self._hash = hash

    __eq__ = slotValuesAreEqual

    def __repr__(self):
        return f"DatasetRef({self.datasetType}, {self.dataId}, id={self.id}, run={self.run})"

    @property
    def id(self):
        """Primary key of the dataset (`int`)

        Typically assigned by `Registry`.
        """
        return self._id

    @property
    def hash(self):
        """Secure hash of the `DatasetType` name and `DataId` (`bytes`).
        """
        if self._hash is None:
            message = hashlib.blake2b(digest_size=32)
            message.update(self.datasetType.name.encode("utf8"))
            self.dataId.fingerprint(message.update)
            self._hash = message.digest()
        return self._hash

    @property
    def datasetType(self):
        """The `DatasetType` associated with the Dataset the `DatasetRef`
        points to.
        """
        return self._datasetType

    @property
    def dataId(self):
        """A mapping of `Dimension` primary key values that labels the Dataset
        within a Collection (`DataCoordinate`).
        """
        return self._dataId

    @property
    def producer(self):
        """The `~lsst.daf.butler.Quantum` instance that produced (or will
        produce) the Dataset.

        Read-only; update via `~lsst.daf.butler.Registry.addDataset()`,
        `~lsst.daf.butler.Quantum.addOutput()`, or
        `~lsst.daf.butler.Butler.put()`.
        May be `None` if no provenance information is available.
        """
        return self._producer

    @property
    def run(self):
        """The `~lsst.daf.butler.Run` instance that produced (or will produce)
        the Dataset.

        Read-only; update via `~lsst.daf.butler.Registry.addDataset()` or
        `~lsst.daf.butler.Butler.put()`.
        """
        return self._run

    @property
    def predictedConsumers(self):
        """A sequence of `Quantum` instances that list this Dataset in their
        `predictedInputs` attributes.

        Read-only; update via `Quantum.addPredictedInput()`.
        May be an empty list if no provenance information is available.
        """
        return _safeMakeMappingProxyType(self._predictedConsumers)

    @property
    def actualConsumers(self):
        """A sequence of `Quantum` instances that list this Dataset in their
        `actualInputs` attributes.

        Read-only; update via `Registry.markInputUsed()`.
        May be an empty list if no provenance information is available.
        """
        return _safeMakeMappingProxyType(self._actualConsumers)

    @property
    def components(self):
        """Named `DatasetRef` components.

        Read-only; update via `Registry.attachComponent()`.
        """
        return _safeMakeMappingProxyType(self._components)

    @property
    def dimensions(self):
        """The dimensions associated with the underlying `DatasetType`
        """
        return self.datasetType.dimensions

    def __str__(self):
        components = ""
        if self.components:
            components = ", components=[" + ", ".join(self.components) + "]"
        return "DatasetRef({}, id={}, dataId={} {})".format(self.datasetType.name,
                                                            self.id, self.dataId, components)

    def detach(self):
        """Obtain a new DatasetRef that is detached from the registry.

        Its ``id`` property will be `None`.  This can be used for transfers
        and similar operations.
        """
        ref = deepcopy(self)
        ref._id = None
        return ref

    def isComponent(self):
        """Boolean indicating whether this `DatasetRef` refers to a
        component of a composite.

        Returns
        -------
        isComponent : `bool`
            `True` if this `DatasetRef` is a component, `False` otherwise.
        """
        return self.datasetType.isComponent()

    def isComposite(self):
        """Boolean indicating whether this `DatasetRef` is a composite type.

        Returns
        -------
        isComposite : `bool`
            `True` if this `DatasetRef` is a composite type, `False`
            otherwise.
        """
        return self.datasetType.isComposite()

    def _lookupNames(self):
        """Name keys to use when looking up this DatasetRef in a configuration.

        The names are returned in order of priority.

        Returns
        -------
        names : `tuple` of `LookupKey`
            Tuple of the `DatasetType` name and the `StorageClass` name.
            If ``instrument`` is defined in the dataId, each of those names
            is added to the start of the tuple with a key derived from the
            value of ``instrument``.
        """
        # Special case the instrument Dimension since we allow configs
        # to include the instrument name in the hierarchy.
        names = self.datasetType._lookupNames()

        if "instrument" in self.dataId:
            names = tuple(n.clone(dataId={"instrument": self.dataId["instrument"]})
                          for n in names) + names

        return names
