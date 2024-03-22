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

# Re-export some top-level exception types for backwards compatibility -- these
# used to be part of registry.
from .._exceptions import DimensionNameError, MissingDatasetTypeError
from .._exceptions_legacy import DataIdError, DatasetTypeError, RegistryError

# Registry imports.
from . import interfaces, managers, queries, wildcards
from ._collection_summary import *
from ._collection_type import *
from ._config import *
from ._defaults import *
from ._exceptions import *
from ._registry import *
from ._registry_factory import *

# Some modules intentionally not imported, either because they are purely
# internal (e.g. nameShrinker.py) or they contain implementations that are
# always loaded from configuration strings (e.g. databases subpackage,
# opaque.py, ...).
