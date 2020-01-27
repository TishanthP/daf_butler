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

"""
Module containing classes used with deferring dataset loading
"""
from __future__ import annotations

__all__ = ("DeferredDatasetHandle",)

import dataclasses
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import DatasetRef
    from .butler import Butler


@dataclasses.dataclass(frozen=True)
class DeferredDatasetHandle:
    """Proxy class that provides deferred loading of a dataset from a butler.
    """

    def get(self, *, parameters: Optional = None, **kwargs: dict) -> Any:
        """ Retrieves the dataset pointed to by this handle

        This handle may be used multiple times, possibly with different
        parameters.

        Parameters
        ----------
        parameters : `dict` or None
            The parameters argument will be passed to the butler get method.
            It defaults to None. If the value is not None,  this dict will
            be merged with the parameters dict used to construct the
            `DeferredDatasetHandle` class.
        kwargs : `dict`
            This argument is deprecated and only exists to support legacy
            gen2 butler code during migration. It is completely ignored
            and will be removed in the future.

        Returns
        -------
        return : `Object`
            The dataset pointed to by this handle
        """
        if self.parameters is not None:
            mergedParameters = self.parameters.copy()
            if parameters is not None:
                mergedParameters.update(parameters)
        elif parameters is not None:
            mergedParameters = parameters
        else:
            mergedParameters = {}

        return self.butler.getDirect(self.ref, parameters=mergedParameters)

    butler: Butler
    """The butler that will be used to fetch the dataset (`Butler`).
    """

    ref: DatasetRef
    """Reference to the dataset (`DatasetRef`).
    """

    parameters: Optional[dict]
    """Optional parameters that may be used to specify a subset of the dataset
    to be loaded (`dict` or `None`).
    """
