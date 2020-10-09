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

from ..utils import MWArgumentDecorator, split_commas


directory_argument = MWArgumentDecorator("directory",
                                         help="DIRECTORY is the folder containing dataset files.")

glob_argument = MWArgumentDecorator("glob",
                                    callback=split_commas,
                                    help="GLOB is one or more strings to apply to the search.",
                                    nargs=-1)

repo_argument = MWArgumentDecorator("repo")

locations_argument = MWArgumentDecorator("locations",
                                         callback=split_commas,
                                         nargs=-1)

dimensions_argument = MWArgumentDecorator("dimensions",
                                          callback=split_commas,
                                          nargs=-1)
