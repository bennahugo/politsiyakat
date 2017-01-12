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
import politsiyakat
import re
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def baseline_index(a1, a2, no_antennae):
    """
     Computes unique index of a baseline given antenna 1 and antenna 2
     (zero indexed) as input. The arrays may or may not contain
     auto-correlations.

     There is a quadratic series expression relating a1 and a2
     to a unique baseline index(can be found by the double difference
     method)

     Let slow_varying_index be S = min(a1, a2). The goal is to find
     the number of fast varying terms. As the slow
     varying terms increase these get fewer and fewer, because
     we only consider unique baselines and not the conjugate
     baselines)
     B = (-S ^ 2 + 2 * S *  # Ant + S) / 2 + diff between the
     slowest and fastest varying antenna

    :param a1: array of ANTENNA_1 ids
    :param a2: array of ANTENNA_2 ids
    :param no_antennae: number of antennae in the array
    :return: array of baseline ids
    """
    if a1.shape != a2.shape:
        raise ValueError("a1 and a2 must have the same shape!")

    slow_index = np.min(np.array([a1, a2]), axis=0)

    return (slow_index * (-slow_index + (2 * no_antennae + 1))) // 2 + \
        np.abs(a1 - a2)

class antenna_tasks:
    """
       Tasks Helper class

       Contains a number of tasks to detect and
       remove / fix cases where antennas have gone bad.
    """

    def __init__(self):
        pass

    @classmethod
    def check_ms(cls, **kwargs):
        """
            Basic ms validity check
        :param_kwargs:
            "msname" : name of measurement set
        """
        try:
            ms = str(kwargs["msname"])
        except:
            raise ValueError("check_ms (or any task that calls it) expects a "
                             "measurement set (key 'ms') as input")

        if not os.path.isdir(ms):
            raise RuntimeError("Measurement set %s does not exist. Check input" % ms)

        with table(ms, readonly=True, ack=False) as t:
            flag_shape = t.getcell("FLAG", 0).shape
            nrows = t.nrows()

        if len(flag_shape) != 2:  # spectral flags are optional in CASA memo 229
            raise RuntimeError("%s does not support storing spectral flags. "
                               "Maybe run pyxis ms.prep?" % ms)
        politsiyakat.log.info("%s appears to be a valid measurement set with %d rows" % (ms, nrows))

    @classmethod
    def flag_excessive_delay_error(cls, **kwargs):
        """
            Flags all those baselines that observed a calibrator field
            with a large portion of timesteps falling outside acceptable phase
            variance. This should catch any antennas that has issues like
            drive problems and digitizer reference timing problems.
        :param kwargs:
            "msname" : name of measurement set
            "DATA" : name of data column
            "cal_field" : calibrator field number (preferably the bandpass calibrator)
            "valid_phase_range" : Phase range (in degrees) specified in the CASA range format
                                  'float~float'
            "max_invalid_timesteps" : Minimum number of timesteps to be invalid before a baseline
                                      is deemed untrustworthy (as % of unflagged data)
            "output_dir" : Where to dump diagnostic plots
        :post-conditions:
            Measurement set is reflagged to invalidate all baselines affected by
            severe phase error.
        """
        antenna_tasks.check_ms(**kwargs)
        ms = str(kwargs["msname"])

        try:
            DATA = str(kwargs["data_column"])
        except:
            raise ValueError("flag_excessive_delay_error expects a data column (key 'data_column') as input")
        try:
            cal_field = int(kwargs["cal_field"])
        except:
            raise ValueError("flag_excessive_delay_error expects a calibrator_field (key 'cal_field') as input")
        try:
            valid_phase_range = kwargs["valid_phase_range"]

            vals = re.match(r"^(?P<lower>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
                            r"~"
                            r"(?P<upper>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$",
                            valid_phase_range)
            if vals is None:
                raise ValueError("Illegal format")
            low_valid_phase = float(vals.group("lower"))
            high_valid_phase = float(vals.group("upper"))
        except:
            raise ValueError("flag_excessive_delay_error expects a valid_phase_range "
                             "(key 'valid_phase_range') as input "
                             "with format 'float~float' in degrees.")
        try:
            max_times = float(kwargs["max_invalid_timesteps"])
        except:
            raise ValueError("flag_excessive_delay_error expects a maximum invalid timesteps (\%) "
                             "(key 'max_invalid_timesteps') as input")

        try:
            output_dir = str(kwargs["output_dir"])
        except:
            raise ValueError("flag_excessive_delay_error expects an output_directory "
                             "(key 'output_dir') as input")

        with table(ms + "::SPECTRAL_WINDOW", readonly=True, ack=False) as t:
            nspw = t.nrows()
            spw_name = t.getcol("NAME")
            spw_nchans = t.getcol("NUM_CHAN")

        assert np.alltrue([spw_nchans[0] == spw_nchans[c] for c in xrange(nspw)]), \
            "for now we can only handle equi-channel spw"
        nchan = spw_nchans[0]

        with table(ms + "::DATA_DESCRIPTION", readonly=True, ack=False) as t:
            map_descriptor_to_spw = t.getcol("SPECTRAL_WINDOW_ID")

        with table(ms + "::ANTENNA", readonly=True, ack=False) as t:
            antenna_names = t.getcol("NAME")
            nant = t.nrows()

        with table(ms + "::POLARIZATION", readonly=True, ack=False) as t:
            ncorr = t.getcol("NUM_CORR")

        assert np.alltrue([ncorr[0] == ncorr[c] for c in xrange(len(ncorr))]), \
            "for now we can only handle rows that all have the same number correlations"
        ncorr = ncorr[0]

        # be conservative autocorrelations is probably still in the mix
        # since they're quite critical in calibration
        no_baselines = (nant * (nant - 1)) // 2 + nant

        # Lets keep a histogram of each channel (all unflagged data)
        # and a corresponding histogram channels where phase is very wrong
        histogram_data = np.zeros([no_baselines, nchan])
        histogram_phase_off = np.zeros([no_baselines, nchan])

        with table(ms, readonly=False, ack=False) as t:
            politsiyakat.log.info("Successfull read-write open of '%s'" % ms)
            nrows_to_read = 1000
            nchunk = int(np.ceil(t.nrows() / float(nrows_to_read)))

            for chunk_i in xrange(nchunk):
                for spw_i in xrange(nspw):
                    politsiyakat.log.info("Computing histogram for chunk %d / %d" % (chunk_i + 1, nchunk))
                    a1 = t.getcol("ANTENNA1",
                                  chunk_i * nrows_to_read,
                                  min(t.nrows() - (chunk_i * nrows_to_read),
                                      nrows_to_read))
                    a2 = t.getcol("ANTENNA2",
                                  chunk_i * nrows_to_read,
                                  min(t.nrows() - (chunk_i * nrows_to_read),
                                      nrows_to_read))
                    baseline = baseline_index(a1, a2, nant)

                    field = t.getcol("FIELD_ID",
                                     chunk_i * nrows_to_read,
                                     min(t.nrows() - (chunk_i * nrows_to_read),
                                         nrows_to_read))
                    data = t.getcol(DATA,
                                    chunk_i * nrows_to_read,
                                    min(t.nrows() - (chunk_i * nrows_to_read),
                                        nrows_to_read))
                    flag = t.getcol("FLAG",
                                    chunk_i * nrows_to_read,
                                    min(t.nrows() - (chunk_i * nrows_to_read),
                                        nrows_to_read))
                    desc = t.getcol("DATA_DESC_ID",
                                    chunk_i * nrows_to_read,
                                    min(t.nrows() - (chunk_i * nrows_to_read),
                                        nrows_to_read))
                    spw = map_descriptor_to_spw[desc]
                    unflagged_data = data * \
                                     np.logical_not(flag) * \
                                     np.tile(field == cal_field,
                                             (ncorr, nchan, 1)).T * \
                                     np.tile(spw == spw_i, (ncorr, nchan, 1)).T

                    histogram_data[baseline, (nchan*spw_i):(nchan * (spw_i + 1))] += \
                        np.array([np.count_nonzero(unflagged_data[:, c, :]) for c in xrange(nchan)])

                    histogram_phase_off[baseline, (nchan*spw_i):(nchan * (spw_i + 1))] += \
                        np.array([np.count_nonzero(np.where((np.angle(unflagged_data[:, c, :]) <
                                                             np.deg2rad(low_valid_phase)) |
                                                            (np.angle(unflagged_data[:, c, :]) >
                                                             np.deg2rad(high_valid_phase))))
                                  for c in xrange(nchan)])

            baseline_flags = np.argwhere(
                histogram_phase_off / (histogram_data + 0.000000001) > (max_times / 100.0))

            print np.count_nonzero(baseline_flags), baseline_flags.size

            for bl, chan in baseline_flags:
                politsiyakat.log.info("Baseline %d channel %d is deemed untrustworthy, will flag it" %
                                      (bl, chan))

            for chunk_i in xrange(nchunk):
                politsiyakat.log.info("Flagging chunk %d / %d" % (chunk_i + 1, nchunk))
                flag = t.getcol("FLAG",
                                chunk_i * nrows_to_read,
                                min(t.nrows() - (chunk_i * nrows_to_read),
                                    nrows_to_read))
                a1 = t.getcol("ANTENNA1",
                              chunk_i * nrows_to_read,
                              min(t.nrows() - (chunk_i * nrows_to_read),
                                  nrows_to_read))
                a2 = t.getcol("ANTENNA2",
                              chunk_i * nrows_to_read,
                              min(t.nrows() - (chunk_i * nrows_to_read),
                                  nrows_to_read))
                baseline = baseline_index(a1, a2, nant)
                for bl, chan in baseline_flags:
                    flag[np.argwhere(baseline == bl), chan % nchan] = True
                t.putcol("FLAG",
                         flag,
                         chunk_i * nrows_to_read,
                         min(t.nrows() - (chunk_i * nrows_to_read),
                             nrows_to_read))

            # Dump a diagnostic plot of the non-flagged data
            fig = plt.figure()
            cm = plt.cm.get_cmap('cubehelix')
            for c in xrange(histogram_data.shape[1]):
                color = [c / float(histogram_data.shape[1])] * histogram_data.shape[0]
                plt.scatter(np.arange(no_baselines),
                            histogram_data[:, c].reshape([no_baselines]),
                            s=1,
                            c=color,
                            cmap=cm)
            plt.title("Unflagged data for " + os.path.basename(ms))
            plt.xlabel("Baseline Index")
            plt.ylabel("Number of unflagged data points")
            fig.savefig(output_dir + "/unflagged_datapoints.field_%d.png" % cal_field)
            plt.close(fig)

            # Dump a diagnostic plot of out of bound phases
            fig = plt.figure()
            cm = plt.cm.get_cmap('cubehelix')
            for c in xrange(histogram_data.shape[1]):
                color = [c / float(histogram_data.shape[1])] * histogram_data.shape[0]
            plt.scatter(np.arange(no_baselines),
                        histogram_phase_off[:, c].reshape([no_baselines]),
                        s=1,
                        c=color,
                        cmap=cm)
            plt.title("Unflagged data for " + os.path.basename(ms))
            plt.xlabel("Baseline Index")
            plt.ylabel("Number of visibilities with invalid phase")
            fig.savefig(output_dir + "/bad_phase_count.field_%d.png" % cal_field)
            plt.close(fig)


