# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 SKA South Africa
#
# This file is part of PolitsiyaKAT.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pyrap.tables import table
import os
import numpy as np

class antenna_tasks:
    """
       Tasks Helper class

       Contains a number of tasks to detect and
       remove / fix cases where antennas have gone bad.
    """

    def __init__(self):
        pass

    @classmethod
    def flag_excessive_delay_error(cls, **kwargs):
        print "stub, msname: %s" % kwargs["msname"]
