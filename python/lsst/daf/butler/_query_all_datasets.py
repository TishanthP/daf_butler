# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This software is dual licensed under the GNU General Public License and also
# under a 3-clause BSD license. Recipients may choose which of these licenses
# to use; please see the files gpl-3.0.txt and/or bsd_license.txt,
# respectively.  If you choose the GPL option then the following text applies
# (but note that there is still no warranty even if you opt for BSD instead):
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

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from lsst.utils.iteration import ensure_iterable

from ._butler import Butler
from ._exceptions import MissingDatasetTypeError
from .dimensions import DataId
from .queries import Query
from .queries.driver import DatasetRefResultPage


class DatasetsPage(NamedTuple):
    dataset_type: str
    data: DatasetRefResultPage


def query_all_datasets(
    self,
    butler: Butler,
    query: Query,
    *,
    collections: str | Iterable[str] | None = None,
    name: str | Iterable[str] = "*",
    find_first: bool = True,
    data_id: DataId | None = None,
    where: str = "",
    bind: Mapping[str, Any] | None = None,
    limit: int | None,
    explain: bool = True,
    **kwargs: Any,
) -> Iterator[DatasetsPage]:
    """Internal implementation method for `Butler.query_all_datasets`.

    Notes
    -----
    ``query_all_datasets`` returns a single list, but for the CLI and the
    Butler server REST API we need to be able to return results one page at a
    time.  So those functions share this common internal method with
    `Butler.query_all_datasets` instead of calling it directly.
    """
    missing_types = []
    dataset_type_query = list(ensure_iterable(name))
    dataset_types = set(butler.registry.queryDatasetTypes(dataset_type_query, missing_types))
    if len(dataset_types) == 0:
        raise MissingDatasetTypeError(f"No dataset types found for query {dataset_type_query}")
    if len(missing_types) > 0:
        raise MissingDatasetTypeError(f"Dataset types not found: {missing_types}")
