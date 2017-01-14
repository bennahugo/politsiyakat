#!/usr/bin/python
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

import os
os.environ['MPLBACKEND'] = "Agg"

import argparse
import logging
import sys
from helpers.antenna_tasks import antenna_tasks
import json
from version import __version__
import politsiyakat

# Where is the module installed?
__install_path = os.path.split(os.path.abspath(politsiyakat.__file__))[0]


def create_logger():
    """ Creates a logger for this module """

    logging.basicConfig(format="%(name)s - %(levelname)s %(asctime)s:\t%(message)s")
    logging.StreamHandler(sys.stdout)
    politsiya_log = logging.getLogger("politsiyakat")
    politsiya_log.setLevel(logging.INFO)
    return politsiya_log

log = create_logger()


def main(argv = None):
    """ Driver: all tasks invoked from here """

    log.info("\n"
             "------------------------------------------------------------------------------------------------------\n"
             "                                                                                                      \n"
             "                                                ,@,                                                   \n"
             "                                               ,@@@,                                                  \n"
             "                                              ,@@@@@,                                                 \n"
             "                                       `@@@@@@@@@@@@@@@@@@@`                                          \n"
             "                                         `@@@@@@@@@@@@@@@`                                            \n"
             "                                           `@@@@@@@@@@@`                                              \n"
             "                                          ,@@@@@@`@@@@@@,                                             \n"
             "                                          @@@@`     `@@@@                                             \n"
             "                                         ;@`           `@;                                            \n"
             " _______  _______  ___      ___   _______  _______  ___   __   __  _______  ___   _  _______  _______ \n"
             "|       ||       ||   |    |   | |       ||       ||   | |  | |  ||   _   ||   | | ||   _   ||       |\n"
             "|    _  ||   _   ||   |    |   | |_     _||  _____||   | |  |_|  ||  |_|  ||   |_| ||  |_|  ||_     _|\n"
             "|   |_| ||  | |  ||   |    |   |   |   |  | |_____ |   | |       ||       ||      _||       |  |   |  \n"
             "|    ___||  |_|  ||   |___ |   |   |   |  |_____  ||   | |_     _||       ||     |_ |       |  |   |  \n"
             "|   |    |       ||       ||   |   |   |   _____| ||   |   |   |  |   _   ||    _  ||   _   |  |   |  \n"
             "|___|    |_______||_______||___|   |___|  |_______||___|   |___|  |__| |__||___| |_||__| |__|  |___|  \n"
             "                                                                                                      \n"
             "------------------------------------------------------------------------------------------------------\n")
    log.info("Module installed at '%s' version %s" % (__install_path, __version__))

    parser = argparse.ArgumentParser(description="Collection of systematic telescope error detection "
                                                 "and mitigation routines")
    parser.add_argument("-s",
                        "--tasksuite",
                        type=str,
                        default="antenna_mod",
                        choices=["antenna_mod"],
                        help="Specify which suite to search for the required task")

    parser.add_argument("task",
                        metavar="task",
                        type=str,
                        help="Name of task to execute, for example flag_excessive_delay_error")

    parser.add_argument("kwargs",
                        metavar="kwargs",
                        default="{}",
                        type=json.loads,
                        help="JSON string containing keyword arguments to the task, for example "
                             "'{\"msname\":\"helloworld.ms\"}'")

    args = parser.parse_args(argv)

    if args.tasksuite == "antenna_mod":
        run_func = getattr(antenna_tasks, args.task, None)
    else:
        raise RuntimeError("Unknown value for taskset. This is a bug.")

    if run_func is None:
        raise RuntimeError("Function %s is not part of suite %s" % (args.task, args.tasksuite))

    log.info("Running task '%s' with the following arguments:" % args.task)
    for (key, val) in args.kwargs.iteritems():
        log.info("\t%s:%s" % (key, val))

    run_func(**args.kwargs)
    log.info("PolitsiyaKAT terminated successfully")
    return 0
