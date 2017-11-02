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
import numpy as np
import politsiyakat
import re
from matplotlib import pyplot as plt
import scipy.interpolate as interp
import matplotlib as mpl
import matplotlib.cm as cm
from politsiyakat.data.misc import *
from politsiyakat.data.data_provider import data_provider
from shared_ndarray import SharedNDArray as sha
import concurrent
import time

class antenna_tasks:
    """
       Tasks Helper class

       Contains a number of tasks to detect and
       remove / fix cases where antennas have gone rogue.
    """

    def __init__(self):
        pass

    @classmethod
    def flag_excessive_delay_error(cls, **kwargs):
        """
            Flags all those baselines that observed a calibrator field
            with a large portion of timesteps falling outside acceptable phase
            variance. This should catch any antennas that has issues like
            drive problems and digitizer reference timing problems.
        :param kwargs:
            "msname" : name of measurement set
            "data_column" : name of data column
            "field" : Comma-seperated list of fields
            "cal_field" : calibrator field number(s), comma-seperated
            "phase_range_clip" : Valid calibrator phase range (in degrees)
                                 specified in the CASA range format
                                  'float~float'
            "invalid_count_frac_clip" : Maximum number of data points (per
                                        correlation per channel per baseline)
                                        to be invalid before a baseline
                                        is deemed untrustworthy (as fraction of
                                        unflagged data for that baseline channel)
            "output_dir" : Where to dump diagnostic plots
            "nrows_chunk" : Number of rows to read per chunk (reduces memory
                            footprint
            "simulate" : Only simulate and compute statistics, don't actually
                         flag anything.
        :post-conditions:
            Measurement set is reflagged to invalidate all baselines affected by
            severe phase error.
        """
        ms_meta = dp.check_ms(**kwargs)
        ms = str(kwargs["msname"])

        try:
            DATA = str(kwargs["data_column"])
        except:
            raise ValueError("flag_excessive_delay_error expects a data column (key 'data_column') as input")
        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["field"]):
                raise ValueError("Expect list of field identifiers")
            fields = [int(f) for f in kwargs["field"].split(",")]
        except:
            raise ValueError("flag_excessive_delay_error expects a field(s) (key "
                             "'field') as input")
        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["cal_field"]):
                raise ValueError("Expect list of field identifiers")
            cal_fields = [int(f) for f in kwargs["cal_field"].split(",")]
        except:
            raise ValueError("flag_excessive_delay_error expects calibrator field(s) (key "
                             "'cal_field') as input")

        if not set(cal_fields).issubset(set(fields)):
            raise ValueError("Calibrator fields must be subset of fields "
                             "that must be flagged.")

        try:
            valid_phase_range = kwargs["phase_range_clip"]

            vals = re.match(r"^(?P<lower>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
                            r"~"
                            r"(?P<upper>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$",
                            valid_phase_range)
            if vals is None:
                raise ValueError("Illegal format")
            low_valid_phase = np.deg2rad(float(vals.group("lower")))
            high_valid_phase = np.deg2rad(float(vals.group("upper")))
        except:
            raise ValueError("flag_excessive_delay_error expects a valid_phase_range "
                             "(key 'phase_range_clip') as input "
                             "with format 'float~float' in degrees.")
        try:
            max_inv_vis = float(kwargs["invalid_count_frac_clip"])
        except:
            raise ValueError("flag_excessive_delay_error expects a fraction of "
                             "maximum invalid visibilities before flagging "
                             "entire baselines from the observation "
                             "(key 'invalid_count_frac_clip') as input")

        try:
            output_dir = str(kwargs["output_dir"])
        except:
            raise ValueError("flag_excessive_delay_error expects an output_directory "
                             "(key 'output_dir') as input")
        try:
            nrows_to_read = int(kwargs["nrows_chunk"])
        except:
            raise ValueError("flag_excessive_delay_error expects number of rows to read per chunk "
                             "(key 'nrows_chunk') as input")
        try:
            simulate = bool(kwargs["simulate"])
            if simulate:
                politsiyakat.log.warn("Warning: you specified you want to "
                                      "simulate a flagging run. This means I "
                                      "will compute statistics for you and "
                                      "dump some diagnostics but not actually "
                                      "touch your data.")

        except:
            raise ValueError("flag_excessive_delay_error expects simulate flag "
                             "(key 'simulate') as input")

        source_names = [ms_meta["ms_field_names"][f] for f in fields]
        nchan = ms_meta["nchan"]
        nspw = ms_meta["nspw"]
        map_descriptor_to_spw = ms_meta["map_descriptor_to_spw"]
        nant = ms_meta["nant"]
        no_fields = len(fields)
        no_baselines = ms_meta["no_baselines"]
        ncorr = ms_meta["ncorr"]
        nchunk = ms_meta["nchunk"]
        antenna_positions = ms_meta["antenna_positions"]
        politsiyakat.log.info("Will flag the following fields:")
        for fi, f in enumerate(fields):
            politsiyakat.log.info("\t(%d): %s" % (f, source_names[fi]) +(
                                  " (calibrator)" if f in cal_fields else ""))

        # Lets keep a histogram of each channel (all unflagged data)
        # and a corresponding histogram channels where phase is very wrong
        histogram_data = np.zeros([no_baselines, nchan * nspw, ncorr],
                                  dtype=np.float64)
        histogram_phase_off = np.zeros([no_baselines, nchan * nspw, ncorr],
                                       dtype=np.float64)
        for chunk_i in xrange(nchunk):
            politsiyakat.log.info("Computing histogram for chunk %d / %d" %
                                  (chunk_i + 1, nchunk))
            politsiyakat.log.info("\tReading MS")
            kwargs["chunk_id"] = chunk_i
            kwargs["ack"] = False
            kwargs["read_exclude"] = ["scan", "time"]
            maintable_chunk = dp.read_ms_maintable_chunk(**kwargs)
            a1 = maintable_chunk["a1"]
            a2 = maintable_chunk["a2"]
            baseline = maintable_chunk["baseline"]
            field = maintable_chunk["field"]
            data = maintable_chunk["data"]
            flag = maintable_chunk["flag"]
            desc = maintable_chunk["desc"]
            spw = maintable_chunk["spw"]

            for spw_i in xrange(nspw):
                in_spw = np.tile(spw == spw_i,
                                 (ncorr, nchan, 1)).T
                for field_i, field_id in enumerate(cal_fields):
                    politsiyakat.log.info("\tProcessing field %s (%d calibrator"
                                          " fields in total)" %
                                          (source_names[fields.index(field_id)],
                                           len(cal_fields)))
                    unflagged_data = data * \
                                     np.logical_not(flag) * \
                                     np.tile(field == field_id,
                                             (ncorr, nchan, 1)).T * \
                                     in_spw

                    # Count all the places where there are 
                    # unflagged correlations
                    S = (unflagged_data != 0.0)
                    # (nrows, nchan)
                    for r in xrange(len(S)):
                        histogram_data\
                            [baseline[r],
                             (nchan*spw_i):(nchan * (spw_i + 1))] +=\
                                S[r]
                    # Where there are some of the correlations outside
                    # valid phase range count, count them
                    ang = np.angle(unflagged_data)
                    less = ang < low_valid_phase
                    more = ang > high_valid_phase
                    L = np.logical_and((np.logical_or(less, more) >
                                        0), S)
                    # (nrows, nchan, ncorr)
                    for r in xrange(len(S)):
                        histogram_phase_off[baseline[r],
                                            (nchan*spw_i):(nchan * (spw_i + 1))] \
                            += L[r]

        # As fraction bigger than tolerated fraction
        F = np.abs(histogram_phase_off /
                   (histogram_data + 0.000000001)) > max_inv_vis
        F *= (histogram_data != 0)
        no_channels_flagged_per_baseline = np.sum(F, axis=1)
        flagged_baseline_channels = np.argwhere(F)
        politsiyakat.log.info("Looking for baselines with general phase error "
                              "across all calibrator fields...")
        flagged_bls_on_phase = False
        for bi in np.argwhere(np.sum(no_channels_flagged_per_baseline, axis=1) > 0):
            politsiyakat.log.info("\tBaseline %d has %s untrustworthy "
                                  "channels per correlation that were not previously "
                                  "flagged." %
                (bi,
                 ",".join([str(cnt) for cnt in
                           no_channels_flagged_per_baseline[bi]])))
            flagged_bls_on_phase = True
        if not flagged_bls_on_phase:
            politsiyakat.log.info("\tDid not find any baselines with "
                                  "significant overall phase drifts "
                                  "when comparing across all "
                                  "calibrator scans")

        for chunk_i in xrange(nchunk):
            politsiyakat.log.info("Applying flags for chunk %d / %d" %
                                  (chunk_i + 1, nchunk))
            politsiyakat.log.info("\tReading MS")
            kwargs["chunk_id"] = chunk_i
            kwargs["ack"] = False
            kwargs["read_exclude"] = ["data", "scan", "time"]
            maintable_chunk = antenna_tasks.read_ms_maintable_chunk(**kwargs)
            a1 = maintable_chunk["a1"]
            a2 = maintable_chunk["a2"]
            baseline = maintable_chunk["baseline"]
            field = maintable_chunk["field"]
            flag = maintable_chunk["flag"]
            desc = maintable_chunk["desc"]
            spw = maintable_chunk["spw"]
            # Apply flags to all fields, including calibrators
            for spw_i in xrange(nspw):
                for field_i, field_id in enumerate(fields):
                    in_spw_field = np.logical_and(spw == spw_i,
                                                  field == field_id)
                    for bl, chan, corr in flagged_baseline_channels:
                        affected_rows = \
                            np.argwhere(np.logical_and(baseline == bl,
                                                       in_spw_field))
                        flag[affected_rows, chan % nchan, corr] = True

            # finally actually touch the measurement set
            if not simulate:
                politsiyakat.log.info("\tWriting flags to MS")
                with table(ms, readonly=False, ack=False) as t:
                    t.putcol("FLAG",
                             flag,
                             chunk_i * nrows_to_read,
                             min(t.nrows() - (chunk_i * nrows_to_read),
                                 nrows_to_read))

        # Dump a diagnostic plot of the number of bad phase channels per
        # baseline
        uv_dist = uv_dist_per_baseline(no_baselines,
                                       nant,
                                       antenna_positions)
        for c in xrange(ncorr):
            fig = plt.figure()
            ranked_uv_dist = np.argsort(uv_dist)
            plt.plot(uv_dist[ranked_uv_dist],
                     no_channels_flagged_per_baseline[ranked_uv_dist, c])
            plt.title(("Flag excessive phase error (corr %d) " % c) + os.path.basename(ms))
            plt.xlabel("UVdist (m)")
            plt.ylabel("Number of bad previously unflagged channels")
            fig.savefig(output_dir + "/%s-FLAGGED_PHASE_UVDIST.OBSWIDE.CORR_%d.png" %
                        (os.path.basename(ms),
                         c))
            plt.close(fig)

    @classmethod
    def flag_autocorr_drifts(cls, **kwargs):
        """
            Flags drifts in the autocorrelations (sky DC). These can be checked prior to 1GC
            Each field is checked independently for drifts in scan averages per antenna. Each antenna is
            additionally checked against a group scan average.
            Averages and variances are computed per channel and correlation.
        :param kwargs:
            "msname" : name of measurement set
            "data_column" : name of data column
            "field" : field number(s) to inspect, comma-seperated
            "cal_field" : calibrator field number(s) to inspect, comma-seperated
            "nrows_chunk" : Number of rows to read per chunk (reduces memory
                            footprint
            "simulate" : Only simulate and compute statistics, don't actually
                         flag anything.

            "output_dir"  : Name of output dir where to dump diagnostic plots
            "scan_to_scan_threshold" : intra antenna scan threshold (antenna-based thresholding)
            "antenna_to_group_threshold" : inter antenna scan threshold (group based thresholding)
        :post-conditions:
            Measurement set is reflagged to invalidate all baselines affected by
            severe phase error.
        """
        ms_meta = data_provider.check_ms(**kwargs)
        ms = str(kwargs["msname"])

        try:
            DATA = str(kwargs["data_column"])
        except:
            raise ValueError("flag_autocorr_drifts expects a data column (key 'data_column') as input")
        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["field"]):
                raise ValueError("Expect list of field identifiers")
            fields = [int(f) for f in kwargs["field"].split(",")]
        except:
            raise ValueError("flag_autocorr_drifts expects field(s) (key "
                             "'field') as input")

        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["cal_field"]):
                raise ValueError("Expect list of field identifiers")
            cal_fields = [int(f) for f in kwargs["cal_field"].split(",")]
        except:
            raise ValueError("flag_autocorr_drifts expects calibrator field(s) (key "
                             "'cal_field') as input")
        try:
            nrows_to_read = int(kwargs["nrows_chunk"])
        except:
            raise ValueError("flag_autocorr_drifts expects number of rows to read per chunk "
                             "(key 'nrows_chunk') as input")
        try:
            simulate = bool(kwargs["simulate"])
            if simulate:
                politsiyakat.log.warn("Warning: you specified you want to "
                                      "simulate a flagging run. This means I "
                                      "will compute statistics for you and "
                                      "dump some diagnostics but not actually "
                                      "touch your data.")

        except:
            raise ValueError("flag_autocorr_drifts expects simulate flag "
                             "(key 'simulate') as input")

        try:
            output_dir = str(kwargs["output_dir"])
        except:
            raise ValueError("flag_autocorr_drifts expects an output_directory "
                             "(key 'output_dir') as input")

        try:
            s2s_sigma = float(kwargs["scan_to_scan_threshold"])
        except:
            raise ValueError("flag_autocorr_drifts expect a scan to scan variation threshold (sigma) "
                             "(key 'scan_to_scan_threshold') as input")

        try:
            a2g_sigma = float(kwargs["antenna_to_group_threshold"])
        except:
            raise ValueError("flag_autocorr_drifts expect a sigma threshold for antenna variation from group"
                             "(key 'antenna_to_group_threshold') as input")

        try:
            dpi = str(kwargs["plot_dpi"])
        except:
            dpi = 600
            politsiyakat.log.warn("Warning: defaulting to plot dpi of 600. Keyword 'plot_dpi' "
                                  "can be used to control this behaviour.")

        try:
            plt_size = str(kwargs["plot_size"])
        except:
            plt_size = 6
            politsiyakat.log.warn("Warning: defaulting to plot size of 6 units. Keyword 'plot_size' can "
                                  "be used to control this behaviour.")

        source_names = [ms_meta["ms_field_names"][f] for f in fields]
        nchan = ms_meta["nchan"]
        antnames = ms_meta["antenna_names"]
        nspw = ms_meta["nspw"]
        map_descriptor_to_spw = ms_meta["map_descriptor_to_spw"]
        nant = ms_meta["nant"]
        no_fields = len(fields)
        no_baselines = ms_meta["no_baselines"]
        ncorr = ms_meta["ncorr"]
        nchunk = ms_meta["nchunk"]
        antenna_positions = ms_meta["antenna_positions"]
        kwargs["ack"] = False
        politsiyakat.log.info("Will process the following fields:")
        for fi, f in enumerate(fields):
            politsiyakat.log.info("\t(%d): %s" % (f, source_names[fi]) + (
                " (calibrator)" if f in cal_fields else ""))

        source_scan_info = {}
        obs_start = None
        with data_provider(msname=ms,
                           data_column=DATA,
                           nrows_chunk=nrows_to_read) as dp:

            # Gather statistics per scan
            for chunk_i, data in enumerate(iter(dp)):
                politsiyakat.log.info("Processing chunk %d of %d..." %
                                      (chunk_i + 1, nchunk))
                politsiyakat.log.info("\tReading MS")
                obs_start = min(obs_start, np.min(data["time"])) \
                    if obs_start is not None else np.min(data["time"])
                politsiyakat.log.info("\tProcessing field:")

                for field_i, field_id in enumerate(fields):
                    if field_id not in source_scan_info:
                        source_scan_info[field_id] = {
                            "is_calibrator": (field_id in cal_fields),
                            "scan_list": [],
                        }
                    source_rows = np.argwhere(data["field"] == field_id)
                    source_scans = np.unique(data["scan"][source_rows])
                    if source_scans.size == 0:
                        politsiyakat.log.info("\t\t\tField %s is not present in this chunk" % source_names[field_id])
                        continue

                    for s in source_scans:
                        scan_rows = np.argwhere(data["scan"] == s)
                        scan_start = np.min(data["time"][scan_rows])
                        scan_end = np.max(data["time"][scan_rows])
                        if s not in source_scan_info[field_id]:
                            source_scan_info[field_id][s] = {
                                "scan_start": scan_start,
                                "scan_end": scan_end,
                                "tot_autopower": np.zeros([nant,
                                                           nchan * nspw,
                                                           ncorr]),
                                "tot_autocount": np.zeros([nant,
                                                          nchan * nspw,
                                                          ncorr]),
                                "tot_flagged": np.zeros([nant,
                                                        nchan * nspw,
                                                        ncorr]),
                                "tot_rowcount": np.zeros([nant,
                                                         nchan * nspw,
                                                         ncorr]),
                                "num_chunks": 1,
                                "chunk_list": set([chunk_i])
                            }
                            source_scan_info[field_id]["scan_list"].append(s)
                        else:
                            source_scan_info[field_id][s]["num_chunks"] += 1
                            source_scan_info[field_id][s]["chunk_list"].add(chunk_i)
                            source_scan_info[field_id][s]["scan_start"] = \
                                min(source_scan_info[field_id][s]["scan_start"], scan_start)
                            source_scan_info[field_id][s]["scan_end"] = \
                                max(source_scan_info[field_id][s]["scan_end"], scan_end)

                        for spw in xrange(nspw):
                            epic_name = 'antenna stats for %s' % source_names[field_id]
                            for a in xrange(nant):
                                politsiyakat.pool.submit_to_epic(epic_name,
                                                                 _wk_per_ant_stats,
                                                                 a,
                                                                 ncorr,
                                                                 nchan,
                                                                 spw,
                                                                 field_id,
                                                                 s,
                                                                 data)

                            res = politsiyakat.pool.collect_epic(epic_name)
                            for r in res:
                                if r[0] == "antstat":
                                    task, ant, \
                                    spw, autopow, \
                                    autocount, totflagged, \
                                    totrowcount = r
                                    source_scan_info[field_id][s]["tot_autopower"][ant, spw * nchan:(spw + 1) * nchan, :] += \
                                        autopow
                                    source_scan_info[field_id][s]["tot_autocount"][ant, spw * nchan:(spw + 1) * nchan, :] += \
                                        autocount
                                    source_scan_info[field_id][s]["tot_flagged"][ant, spw * nchan:(spw + 1) * nchan, :] += \
                                        totflagged
                                    source_scan_info[field_id][s]["tot_rowcount"][ant, spw * nchan:(spw + 1) * nchan, :] += \
                                        totrowcount
                    tot_flagged = 0
                    tot_sel = 0
                    for s in source_scans:
                        tot_flagged += np.sum(source_scan_info[field_id][s]["tot_flagged"])
                        tot_sel += np.sum(source_scan_info[field_id][s]["tot_rowcount"])
                    if tot_sel != 0:
                        politsiyakat.log.info("\t\t\tField %s is %.2f %% flagged in this chunk" %
                                              (source_names[field_id],
                                               tot_flagged / tot_sel * 100.0))

            # Print some stats per field
            politsiyakat.log.info("Summary of flagging statistics per field:")
            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tField %s has the following scans:" % source_names[field_id])
                for s in source_scan_info[field_id]["scan_list"]:
                    flagged = np.sum(source_scan_info[field_id][s]["tot_flagged"])
                    count = np.sum(source_scan_info[field_id][s]["tot_rowcount"])
                    politsiyakat.log.info("\t\tScan %d (duration: %0.2f seconds) is %.2f %% flagged" %
                                          (s,
                                           source_scan_info[field_id][s]["scan_end"] -
                                           source_scan_info[field_id][s]["scan_start"],
                                           flagged / count * 100.0
                                           ))
            # Print some stats per antenna
            ant_flagged = np.zeros([nant])
            ant_count = np.zeros([nant])
            for field_i, field_id in enumerate(fields):
                for s in source_scan_info[field_id]["scan_list"]:
                        ant_flagged += np.sum(np.sum(source_scan_info[field_id][s]["tot_flagged"],
                                                     axis=1),
                                              axis=1)
                        ant_count += np.sum(np.sum(source_scan_info[field_id][s]["tot_rowcount"],
                                                     axis=1),
                                              axis=1)

            politsiyakat.log.info("Flagging statistics per antenna:")
            for ant in xrange(nant):
                politsiyakat.log.info("\tAntenna %s is %.2f %% flagged" %
                                      (antnames[ant], ant_flagged[ant] / ant_count[ant] * 100.0))

            # Compute median amplitude band per antenna per field
            ant_median_amp = np.zeros([no_fields, nant, nchan*nspw, ncorr])
            ant_std_amp = np.zeros([no_fields, nant, nchan * nspw, ncorr])
            field_scan_flags_intra = {}
            field_scan_flags_inter = {}
            for field_i, field_id in enumerate(fields):
                # flag individual channels from scans exceeding sigma tolerance comparing intra-antenna
                amp_scans = np.zeros([len(source_scan_info[field_id]["scan_list"]), nant, nchan * nspw, ncorr])
                scan_amp_flags = np.zeros([len(source_scan_info[field_id]["scan_list"]), nant, nchan * nspw, ncorr],
                                          np.bool)
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    amp_scans[si] = np.divide(source_scan_info[field_id][s]["tot_autopower"],
                                              source_scan_info[field_id][s]["tot_rowcount"])
                ant_median_amp[field_i] = np.nanmean(amp_scans, axis=0)
                ant_std_amp[field_i] = np.nanstd(amp_scans, axis=0)
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    scan_amp_flags[si] = np.abs(amp_scans[si] - ant_median_amp[field_i]) > s2s_sigma * ant_std_amp[field_i]
                field_scan_flags_intra[field_id] = scan_amp_flags

                # flag individual channels from scans exceeding sigma tolerance comparing inter-antenna
                scan_amp_flags_inter = np.zeros([len(source_scan_info[field_id]["scan_list"]), nant, nchan * nspw, ncorr],
                                                np.bool)
                median_array = np.nanmedian(ant_median_amp[field_i], axis=0)
                std_array = np.nanstd(ant_median_amp[field_i], axis=0)
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    scan_amp_flags_inter[si] = np.abs(amp_scans[si] - median_array) > a2g_sigma * std_array
                field_scan_flags_inter[field_id] = scan_amp_flags_inter

            # Print new intra antanna flagging statistics
            politsiyakat.log.info("Resulting intra-antenna scan flagging based on autocorraltion amplitude:")
            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tField %s:" % source_names[field_id])
                for ant in xrange(nant):
                    politsiyakat.log.info("\t\tAntenna %s:" % antnames[ant])
                    for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                        flagged = np.sum(field_scan_flags_intra[field_id][si][ant], axis=0)
                        count = nchan*nspw
                        politsiyakat.log.info("\t\t\tCorrelations of scan %d will contain [%s] %% flagged channels per row according to criterion" %
                                              (s,
                                               ','.join(["%.2f" % v for v in flagged / float(count) * 100.0])))

            # Print new inter antanna flagging statistics
            politsiyakat.log.info(
                "Resulting inter-antenna scan flagging based on autocorraltion amplitude:")
            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tField %s:" % source_names[field_id])
                for ant in xrange(nant):
                    politsiyakat.log.info("\t\tAntenna %s:" % antnames[ant])
                    for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                        flagged = np.sum(field_scan_flags_inter[field_id][si][ant], axis=0)
                        count = nchan * nspw
                        politsiyakat.log.info(
                            "\t\t\tCorrelations of scan %d will contain [%s] %% flagged channels per row according to criterion" %
                            (s,
                             ','.join(["%.2f" % v for v in flagged / float(count) * 100.0])))

            # Write back per-antenna scan-based flags:
            dp.read_exclude = ['data', 'time']
            for chunk_i, data in enumerate(iter(dp)):
                politsiyakat.log.info("Updating flags for chunk %d of %d..." %
                                      (chunk_i + 1, nchunk))
                politsiyakat.log.info("\tReading MS")
                for field_i, field_id in enumerate(source_scan_info.keys()):
                    politsiyakat.log.info("\tSelecting field %s..." %
                                          source_names[field_id])
                    has_updated = False
                    for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                        if chunk_i not in source_scan_info[field_id][s]["chunk_list"]:
                            continue
                        has_updated = True
                        politsiyakat.log.info("\t\tUpdating flags for scan %d" % s)
                        # per antenna, all spw one go
                        for a in xrange(nant):
                            epic_name = 'antenna flag update for %s' % source_names[field_id]
                            politsiyakat.pool.submit_to_epic(epic_name,
                                                             _wk_per_ant_update,
                                                             a,
                                                             field_id,
                                                             s,
                                                             si,
                                                             data,
                                                             field_scan_flags_intra,
                                                             field_scan_flags_inter)
                        politsiyakat.pool.collect_epic(epic_name)
                    if not has_updated:
                        politsiyakat.log.info("\t\tNothing to be done for this field")
                if not simulate:
                    politsiyakat.log.info("\t\tWriting flag buffer back to disk...")
                    dp.flush_flags()


            # Create waterfall plots
            politsiyakat.log.info("Creating waterfall plots:")
            politsiyakat.log.info("\tInterpolating onto a common axis...")
            obs_start = np.inf
            obs_end = -np.inf
            for field_i, field_id in enumerate(fields):
                for s in source_scan_info[field_id]["scan_list"]:
                    obs_end = max(obs_end,
                        source_scan_info[field_id][s]["scan_end"])
                    obs_start = min(obs_start,
                        source_scan_info[field_id][s]["scan_start"])
            try:
                heatmaps = sha([len(fields), nant, ncorr, 512, nchan * nspw])
                famp = {}
                for field_i, field_id in enumerate(fields):
                    sh = [len(source_scan_info[field_id]["scan_list"]), nant, nchan * nspw, ncorr]

                    if np.prod(np.array(sh)) == 0:
                        continue # nothing to do
                    famp[field_i] = sha(sh)
                    scan_mid =  np.zeros([len(source_scan_info[field_id]["scan_list"])])
                    for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                        newf = np.logical_or(field_scan_flags_intra[field_id][si],
                                             field_scan_flags_inter[field_id][si])
                        famp[field_i].array[si, :, :, :] = \
                            np.divide(source_scan_info[field_id][s]["tot_autopower"],
                                      source_scan_info[field_id][s]["tot_rowcount"])
                        famp[field_i].array[si][newf] = np.nan
                        scan_mid[si] = np.array([0.5 * (source_scan_info[field_id][s]["scan_end"] -
                                                        source_scan_info[field_id][s]["scan_start"]) - obs_start])

                    for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                        for ant in xrange(nant):
                            politsiyakat.pool.submit_to_epic("waterfall plots",
                                                             _wkr_ant_corr_regrid,
                                                             source_names[field_id],
                                                             field_i,
                                                             ncorr,
                                                             nchan,
                                                             nspw,
                                                             obs_end,
                                                             obs_start,
                                                             scan_mid,
                                                             famp,
                                                             antnames,
                                                             heatmaps,
                                                             ant)
                    politsiyakat.pool.collect_epic("waterfall plots")
                    politsiyakat.log.info("\t\t Done...")
                    for corr in xrange(ncorr):
                        nxplts = int(np.ceil(np.sqrt(nant)))
                        nyplts = int(np.ceil(nant / nxplts))
                        f, axarr = plt.subplots(nyplts, nxplts, dpi=dpi, figsize=(nyplts*plt_size,
                                                                                  nxplts*plt_size))
                        dbheatmaps = 10 * np.log10(heatmaps.array[:, :, :, :, :])
                        #sanatize
                        dbheatmaps[dbheatmaps == -np.inf] = np.nan
                        dbheatmaps[dbheatmaps == np.inf] = np.nan
                        scale_min = np.nanmin(dbheatmaps[field_id, :, corr, :, :])
                        scale_max = np.nanmax(dbheatmaps[field_id, :, corr, :, :])
                        for ant in xrange(nant):
                            dbscale = dbheatmaps[field_id, ant, corr, :, :]

                            im = axarr[ant // nxplts, ant % nxplts].imshow(
                                dbscale,
                                aspect = 'auto',
                                extent = [0, nchan * nspw,
                                          0, (obs_end - obs_start) / 3600.0],
                                vmin = scale_min,
                                vmax = scale_max)
                            plt.colorbar(im, ax = axarr[ant // nxplts, ant % nxplts])
                            axarr[ant // nxplts, ant % nxplts].set_title(antnames[ant])
                        f.savefig(output_dir + "/%s-AUTOCORR-FIELD-%s-CORR-%d.png" %
                                    (os.path.basename(ms),
                                     source_names[field_id],
                                     corr))
                        plt.close(f)
                politsiyakat.log.info("\t\t Saved to %s" % output_dir)
            finally:
                heatmaps.unlink()
                for a in famp.keys():
                    famp[a].unlink()

def _wk_per_ant_update(a,
                       field_id,
                       s,
                       si,
                       data,
                       field_scan_flags_intra,
                       field_scan_flags_inter):
    nrow_per_chunk = 100
    nchunks = int(np.ceil(data["data"].shape[0] / float(nrow_per_chunk)))
    for n in xrange(nchunks):
        nrows_to_read = min(data["data"].shape[0] - (n * nrow_per_chunk),
                            nrow_per_chunk)
        chunk_start = n * nrow_per_chunk
        chunk_end = n * nrow_per_chunk + nrows_to_read
        field_sel = data["field"][chunk_start:chunk_end] == field_id
        scan_sel = data["scan"][chunk_start:chunk_end] == s
        accum_sel = np.logical_and(field_sel, scan_sel)
        ant_sel = np.logical_or(data["a1"][chunk_start:chunk_end] == a,
                                data["a2"][chunk_start:chunk_end] == a)
        accum_sel = np.logical_and(accum_sel, ant_sel)
        newf = np.logical_or(field_scan_flags_intra[field_id][si][a],
                             field_scan_flags_inter[field_id][si][a])
        data["flag"][np.argwhere(accum_sel)][chunk_start:chunk_end] = \
            np.logical_or(data["flag"][np.argwhere(accum_sel)][chunk_start:chunk_end],
                          newf)

def _wk_per_ant_stats(a,
                      ncorr,
                      nchan,
                      spw,
                      field_id,
                      s,
                      data):
    # get autocorrelation power per antenna
    autopow = None
    autocount = None
    totflagged = None
    totrowcount = None
    nrow_per_chunk = 100
    nchunks = int(np.ceil(data["data"].shape[0] / float(nrow_per_chunk)))
    for n in xrange(nchunks):
        nrows_to_read = min(data["data"].shape[0] - (n * nrow_per_chunk),
                            nrow_per_chunk)
        chunk_start = n*nrow_per_chunk
        chunk_end = n*nrow_per_chunk + nrows_to_read

        field_sel = data["field"][chunk_start:chunk_end] == field_id
        scan_sel = data["scan"][chunk_start:chunk_end] == s
        accum_sel = np.logical_and(field_sel, scan_sel)
        spw_sel = data["spw"][chunk_start:chunk_end] == spw
        accum_sel = np.logical_and(accum_sel, spw_sel)
        accum_sel_tiled = np.tile(accum_sel, (ncorr, nchan, 1)).T
        autocorrs_sel = np.logical_and(data["a1"][chunk_start:chunk_end] == data["a2"][chunk_start:chunk_end],
                                       data["a1"][chunk_start:chunk_end] == a)
        sel = np.logical_and(accum_sel_tiled,
                             np.logical_and(np.logical_not(data["flag"][chunk_start:chunk_end]),
                                            np.tile(autocorrs_sel,
                                                    (ncorr, nchan, 1)).T))


        dn = data["data"][chunk_start:chunk_end].copy()
        dn[np.logical_not(sel)] = np.nan
        d = np.abs(dn)
        f = np.logical_and(data["flag"][chunk_start:chunk_end],
                           sel)
        if autopow is None:
            autopow = np.nansum(d, axis=0)
        else:
            autopow += np.nansum(d, axis=0)
        if autocount is None:
            autocount = np.sum(f, axis=0)
        else:
            autocount += np.sum(f, axis=0)

        # get flagging statistics per antenna
        antsel = np.logical_or(data["a1"][chunk_start:chunk_end] == a,
                               data["a2"][chunk_start:chunk_end] == a)
        sel = np.tile(np.logical_and(accum_sel,
                                     antsel),
                      (ncorr, nchan, 1)).T
        f = np.logical_and(data["flag"][chunk_start:chunk_end], sel)
        if totflagged is None:
            totflagged = np.sum(f, axis=0)
        else:
            totflagged += np.sum(f, axis=0)
        if totrowcount is None:
            totrowcount = np.sum(sel, axis=0)
        else:
            totrowcount += np.sum(sel, axis=0)
    return ("antstat", a, spw, autopow, autocount, totflagged, totrowcount)

def _wkr_ant_corr_regrid(field_name,
                         field_i,
                         ncorr,
                         nchan,
                         nspw,
                         obs_end,
                         obs_start,
                         scan_mid,
                         famp,
                         antnames,
                         heatmaps,
                         ant):
    for corr in xrange(ncorr):
        X, Y = np.linspace(0.0, float(obs_end - obs_start), 512), \
               np.arange(nchan * nspw)
        xcoords = np.repeat(scan_mid, nchan * nspw)
        ycoords = np.tile(np.arange(nchan * nspw),
                          [1, famp[field_i].array.shape[0]]).flatten()
        heatmap = interp.griddata((xcoords,
                                   ycoords),
                                  famp[field_i].array[:, ant, :, corr].flatten(),
                                  (X[:, None], Y[None, :]),
                                  method='nearest')
        heatmaps.array[field_i, ant, corr, :, :] = heatmap