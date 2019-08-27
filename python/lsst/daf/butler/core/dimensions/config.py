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

__all__ = ("DimensionConfig", )

from typing import Tuple, Dict

from ..config import Config, ConfigSubset
from ..utils import doImport
from ..schema import FieldSpec
from .elements import DimensionElement, Dimension, SkyPixDimension


class DimensionConfig(ConfigSubset):
    """Configuration that defines a `DimensionUniverse`.

    The configuration tree for dimensions is a (nested) dictionary
    with four top-level entries:

    - version: an integer version number, used as keys in a singleton registry
      of all `DimensionUniverse` instances;

    - skypix: a dictionary whose entries each define a `SkyPixDimension`,
      along with a special "common" key whose value is the name of a skypix
      dimension that is used to relate all other spatial dimensions in the
      `Registry` database;

    - elements: a nested dictionary whose entries each define a non-skypix
      `DimensionElement`;

    - packers: a nested dictionary whose entries define a factory for a
      `DimensionPacker` instance.
    """
    component = "dimensions"
    requiredKeys = ("version", "elements", "skypix")
    defaultConfigFile = "dimensions.yaml"


def processSkyPixConfig(config: Config) -> Tuple[Dict[str, SkyPixDimension], SkyPixDimension]:
    """Process the "skypix" section of a `DimensionConfig`.

    Parameters
    ----------
    config : `Config`
        The subset of a `DimensionConfig` that corresponds to the "skypix" key.

    Returns
    -------
    dimensions: `dict`
        A dictionary mapping `str` names to partially-constructed
        `SkyPixDimension` instances; the called (i.e. a `DimensionUniverse`)
        is responsible for calling `DimensionElement._finish` to complete
        construction.
    common: `SkyPixDimension`
        The special dimension used to relate all other spatial dimensions in
        the universe.  This instance is also guaranteed to be a value in
        the returned ``dimensions``.
    """
    skyPixNames = set(config.keys())
    skyPixNames.remove("common")
    dimensions = {}
    for name in skyPixNames:
        subconfig = config[name]
        pixelizationClass = doImport(subconfig["class"])
        level = subconfig.get("level", None)
        if level is not None:
            pixelization = pixelizationClass(level)
        else:
            pixelization = pixelizationClass()
        dimensions[name] = SkyPixDimension(name, pixelization)
    return dimensions, dimensions[config["common"]]


def processElementsConfig(config: Config) -> Dict[str, DimensionElement]:
    """Process the "elements" section of a `DimensionConfig`.

    Parameters
    ----------
    config : `Config`
        The subset of a `DimensionConfig` that corresponds to the "elements"
        key.

    Returns
    -------
    dimensions : `dict`
        A dictionary mapping `str` names to partially-constructed
        `DimensionElement` instances; the called (i.e. a `DimensionUniverse`)
        is responsible for calling `DimensionElement._finish` to complete
        construction.
    """
    elements = dict()
    for name, subconfig in config.items():
        kwargs = {}
        kwargs["impliedDependencyNames"] = frozenset(subconfig.get("implies", ()))
        kwargs["directDependencyNames"] = \
            kwargs["impliedDependencyNames"].union(subconfig.get("requires", ()))
        kwargs["metadata"] = [FieldSpec.fromConfig(c) for c in subconfig.get("metadata", ())]
        kwargs["spatial"] = subconfig.get("spatial", False)
        kwargs["temporal"] = subconfig.get("temporal", False)
        kwargs["cached"] = subconfig.get("cached", False)
        kwargs["viewOf"] = subconfig.get("view_of", None)
        keys = subconfig.get("keys")
        if keys is not None:
            uniqueKeys = [FieldSpec.fromConfig(c, nullable=False) for c in keys]
            uniqueKeys[0].primaryKey = True
            elements[name] = Dimension(name, uniqueKeys=uniqueKeys, **kwargs)
        else:
            elements[name] = DimensionElement(name, **kwargs)
    return elements
