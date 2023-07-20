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

import logging
from collections.abc import Iterable
from types import EllipsisType
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table as AstropyTable

from .._butler import Butler, DataCoordinate
from ..cli.utils import sortAstropyTable

if TYPE_CHECKING:
    from lsst.daf.butler import DimensionGraph

_LOG = logging.getLogger(__name__)


class _Table:
    """Aggregates DataIds and creates an astropy table with one DataId per
    row. Eliminates duplicate rows.

    Parameters
    ----------
    dataIds : `iterable` [ ``DataId`` ]
        The DataIds to add to the table.
    """

    def __init__(self, dataIds: Iterable[DataCoordinate]):
        # use dict to store dataIds as keys to preserve ordering
        self.dataIds = dict.fromkeys(dataIds)

    def getAstropyTable(self, order: bool) -> AstropyTable:
        """Get the table as an astropy table.

        Parameters
        ----------
        order : `bool`
            If True then order rows based on DataIds.

        Returns
        -------
        table : `astropy.table.Table`
            The dataIds, sorted by spatial and temporal columns first, and then
            the rest of the columns, with duplicate dataIds removed.
        """
        # Should never happen; adding a dataset should be the action that
        # causes a _Table to be created.
        if not self.dataIds:
            raise RuntimeError("No DataIds were provided.")

        dataId = next(iter(self.dataIds))
        dimensions = list(dataId.full.keys())
        columnNames = [str(item) for item in dimensions]

        # Need to hint the column types for numbers since the per-row
        # constructor of Table does not work this out on its own and sorting
        # will not work properly without.
        typeMap = {float: np.float64, int: np.int64}
        columnTypes = [typeMap.get(type(value)) for value in dataId.full.values()]

        rows = [[value for value in dataId.full.values()] for dataId in self.dataIds]

        table = AstropyTable(np.array(rows), names=columnNames, dtype=columnTypes)
        if order:
            table = sortAstropyTable(table, dimensions)
        return table


def queryDataIds(
    repo: str,
    dimensions: Iterable[str],
    datasets: tuple[str, ...],
    where: str,
    collections: Iterable[str],
    order_by: tuple[str, ...],
    limit: int,
    offset: int,
) -> tuple[AstropyTable | None, str | None]:
    """Query for data IDs.

    Docstring for supported parameters is the same as
    `~lsst.daf.butler.Registry.queryDataIds`.
    """
    butler = Butler(repo, skip_datastore=True)

    if datasets and collections and not dimensions:
        # Determine the dimensions relevant to all given dataset types.
        # Since we are going to AND together all dimensions, we can not
        # seed the result with an empty set.
        graph: DimensionGraph | None = None
        dataset_types = list(butler.registry.queryDatasetTypes(datasets))
        for dataset_type in dataset_types:
            if graph is None:
                # Seed with dimensions of first dataset type.
                graph = dataset_type.dimensions
            else:
                # Only retain dimensions that are in the current
                # set AND the set from this dataset type.
                graph = graph.intersection(dataset_type.dimensions)
            _LOG.debug("Dimensions now %s from %s", set(graph.names), dataset_type.name)

            # Break out of the loop early. No additional dimensions
            # can be added to an empty set when using AND.
            if not graph:
                break

        if not graph:
            names = [d.name for d in dataset_types]
            return None, f"No dimensions in common for specified dataset types ({names})"
        dimensions = set(graph.names)
        _LOG.info("Determined dimensions %s from datasets option %s", dimensions, datasets)

    query_collections: Iterable[str] | EllipsisType | None = None
    if datasets:
        query_collections = collections if collections else ...
    results = butler.registry.queryDataIds(
        dimensions, datasets=datasets, where=where, collections=query_collections
    )

    if order_by:
        results = results.order_by(*order_by)
    if limit > 0:
        new_offset = offset if offset > 0 else None
        results = results.limit(limit, new_offset)

    if results.any(exact=False):
        if results.graph:
            table = _Table(results)
            if not table.dataIds:
                return None, "Post-query region filtering removed all rows, since nothing overlapped."
            return table.getAstropyTable(not order_by), None
        else:
            return None, "Result has one logical row but no columns because no dimensions were requested."
    else:
        return None, "\n".join(results.explain_no_results())
