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
from politsiyakat.modules.flag_tasks import flag_tasks
import json
import pkg_resources
try:
    __version__ = pkg_resources.require("politsiyakat")[0].version
except pkg_resources.DistributionNotFound:
    __version__ = "dev"

import politsiyakat
from politsiyakat.processing.async_pool import async_pool

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
pool = None

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
                        help="Name of task to execute, for example flag_phase_drifts")

    parser.add_argument("kwargs",
                        metavar="kwargs",
                        default="{}",
                        type=json.loads,
                        help="JSON string containing keyword arguments to the task, for example "
                             "'{\"msname\":\"helloworld.ms\"}'")

    args = parser.parse_args(argv)
    try:
        nio_threads = args.kwargs.pop("nio_threads")
    except:
        politsiyakat.log.warn("Setting maximum number of IO threads to 1. This behaviour can be controlled "
                              "via parameter 'nio_threads'")
        nio_threads = 1
    try:
        nproc_threads = args.kwargs.pop("nproc_threads")
    except:
        politsiyakat.log.warn("Setting maximum number of worker threads to 1. This behaviour can be controlled "
                              "via parameter 'nproc_threads'")
        nproc_threads = 1
    global pool
    pool = async_pool(nio_threads, nproc_threads)

    if args.tasksuite == "antenna_mod":
        run_func = getattr(flag_tasks, args.task, None)
    else:
        raise RuntimeError("Unknown value for taskset. This is a bug.")

    if run_func is None:
        raise RuntimeError("Function %s is not part of suite %s" % (args.task, args.tasksuite))

    log.info("Running task '%s' with the following arguments:" % args.task)
    for (key, val) in args.kwargs.iteritems():
        log.info("\t%s:%s" % (key, val))
    try:
        run_func(**args.kwargs)
        return 0
    finally:
        log.info("Waiting for remaining jobs to finish...")
        pool.shutdown()


