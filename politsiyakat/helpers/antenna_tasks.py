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
import matplotlib as mpl
import matplotlib.cm as cm
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

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

def uv_dist_per_baseline(no_baselines, nant, antenna_positions):
    """
    Given antenna positions in an earth centric earth fixed frame
    compute the unscaled uv distance per baseline
    :param no_baselines: Number of baselines
    :param nant: Number of antennae
    :param antenna_positions: array of [X,Y,Z] ITRS antenna positions
                              as found in a MS ANTENNA table
    return array of uv distances per unique baseline index
    """
    # Antenna positions should be in some earth centric earth fixed frame
    # which can be rotated to celestial horizon and then to uvw coordinates
    # so I'm assuming this will be an okay estimate for uv dist (with no
    # scaling by wavelength)
    uv_dist_sq = np.zeros([no_baselines])
    relative_position_a0 = antenna_positions - antenna_positions[0]
    lbound = 0
    bi = 0
    for a0 in xrange(lbound, nant):
        for a1 in xrange(a0, nant):
            uv_dist_sq[bi] = np.sum((relative_position_a0[a0] -
                                     relative_position_a0[a1]) ** 2)
            bi += 1
        lbound += 1
    return np.sqrt(uv_dist_sq)


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
            Basic ms validity check and meta data extraction
        :param_kwargs:
            "msname" : name of measurement set
            "nrows_chunk" : number of rows per chunk
        """
        try:
            ms = str(kwargs["msname"])
        except:
            raise ValueError("check_ms (or any task that calls it) expects a "
                             "measurement set (key 'msname') as input")

        try:
            nrows_to_read = int(kwargs["nrows_chunk"])
        except:
            raise ValueError("Task check_ms expects num "
                             "rows per chunk (key 'nrows_chunk')")
        try:
            ack = kwargs["ack"]
        except:
            ack = True

        if not os.path.isdir(ms):
            raise RuntimeError("Measurement set %s does not exist. Check input" % ms)

        ms_meta = {}

        with table(ms, readonly=True, ack=False) as t:
            ms_meta["nchunk"] = int(np.ceil(t.nrows() /
                                            float(nrows_to_read)))
            flag_shape = t.getcell("FLAG", 0).shape
            ms_meta["nrows"] = t.nrows()

        if len(flag_shape) != 2:  # spectral flags are optional in CASA memo 229
            raise RuntimeError("%s does not support storing spectral flags. "
                               "Maybe run pyxis ms.prep?" % ms)

        with table(ms + "::FIELD", readonly=True, ack=False) as t:
            ms_meta["ms_field_names"] = t.getcol("NAME")

        with table(ms + "::SPECTRAL_WINDOW", readonly=True, ack=False) as t:
            ms_meta["nspw"] = t.nrows()
            ms_meta["spw_name"] = t.getcol("NAME")
            spw_nchans = t.getcol("NUM_CHAN")

        assert np.alltrue([spw_nchans[0] == spw_nchans[c]
                           for c in xrange(ms_meta["nspw"])]), \
            "for now we can only handle equi-channel spw"
        ms_meta["nchan"] = spw_nchans[0]

        with table(ms + "::DATA_DESCRIPTION", readonly=True, ack=False) as t:
            ms_meta["map_descriptor_to_spw"] = t.getcol("SPECTRAL_WINDOW_ID")

        with table(ms + "::ANTENNA", readonly=True, ack=False) as t:
            ms_meta["antenna_names"] = t.getcol("NAME")
            ms_meta["antenna_positions"] = t.getcol("POSITION")
            ms_meta["nant"] = t.nrows()

        # be conservative autocorrelations is probably still in the mix
        # since they can be quite critical in calibration
        ms_meta["no_baselines"] = \
            (ms_meta["nant"] * (ms_meta["nant"] - 1)) // 2 + ms_meta["nant"]

        with table(ms + "::POLARIZATION", readonly=True, ack=False) as t:
            ncorr = t.getcol("NUM_CORR")
        assert np.alltrue([ncorr[0] == ncorr[c] for c in xrange(len(ncorr))]), \
            "for now we can only handle rows that all have the same number correlations"
        ms_meta["ncorr"] = ncorr[0]
        if ack:
            politsiyakat.log.info("%s appears to be a valid measurement set with %d rows" % 
                                  (ms, ms_meta["nrows"]))
        return ms_meta

    @classmethod
    def read_ms_maintable_chunk(cls, **kwargs):
        """
            Reads and returns a given chunk of data
        :param kwargs:
            "msname" : name of measurement set
            "data_column" : data column to use for amplitude flagging
            "chunk_id" : id of chunk to read
            "nrows_chunk" : size of chunk in rows to read
        """
        ms_meta = antenna_tasks.check_ms(**kwargs)
        ms = str(kwargs["msname"])
        try:
            msname = str(kwargs["msname"])
        except:
            raise ValueError("Task read_ms_maintable_chunk expects ms name "
                             "(key 'msname')")
        try:
            chunk_i = int(kwargs["chunk_id"])
        except:
            raise ValueError("Task read_ms_maintable_chunk expects chunk id "
                             "(key 'chunk_id')")
        try:
            nrows_to_read = int(kwargs["nrows_chunk"])
        except:
            raise ValueError("Task read_ms_maintable_chunk expects num "
                             "rows per chunk (key 'nrows_chunk')")
        try:
            DATA = str(kwargs["data_column"])
        except:
            raise ValueError("flag_amplitude_drifts expects a data column (key 'data_column') as input")
        try:
            if not isinstance(kwargs["read_exclude"], list):
                raise ValueError("readexclude list not list")
            read_exclude = kwargs["read_exclude"]
        except:
            read_exclude = []

        maintable_chunk = {}
        map_descriptor_to_spw = ms_meta["map_descriptor_to_spw"]

        with table(ms, readonly=True, ack=False) as t:
            if not "a1" in read_exclude:
                maintable_chunk["a1"] = t.getcol(
                    "ANTENNA1",
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))
            if not "a2" in read_exclude:
                maintable_chunk["a2"] = t.getcol(
                    "ANTENNA2",
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))
            if not "a1" in read_exclude \
               and not "a2" in read_exclude:
                maintable_chunk["baseline"] = \
                    baseline_index(maintable_chunk["a1"],
                                   maintable_chunk["a2"],
                                   ms_meta["nant"])
            if not "field" in read_exclude:
                maintable_chunk["field"] = t.getcol(
                    "FIELD_ID",
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))
            if not "data" in read_exclude:
                maintable_chunk["data"] = t.getcol(
                    DATA,
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))
            if not "flag" in read_exclude:
                maintable_chunk["flag"] = t.getcol(
                    "FLAG",
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))
            if not "desc" in read_exclude:
                maintable_chunk["desc"] = t.getcol(
                    "DATA_DESC_ID",
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))
            if not "spw" in read_exclude\
               and not "desc" in read_exclude:
                maintable_chunk["spw"] = \
                    map_descriptor_to_spw[maintable_chunk["desc"]]
            if not "scan" in read_exclude:
                maintable_chunk["scan"] = t.getcol(
                    "SCAN_NUMBER",
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))
            if not "time" in read_exclude:
                maintable_chunk["time"] = t.getcol(
                    "TIME",
                    chunk_i * nrows_to_read,
                    min(t.nrows() - (chunk_i * nrows_to_read),
                        nrows_to_read))

        return maintable_chunk

    @classmethod
    def flag_excessive_amp_phase_error(cls, **kwargs):
        """
            Flags several problems:
                1. Visibility amplitudes of baselines
                   (per channel and correlation) that drift in time
                   (comparing mean power in each scan to other scans in the field).
                2. (Calibrators only) Phases of calibrators that fall
                   outside specified clip range, per channel and correlation
                3. (Calibrators only) Baselines (per channel, correlation) that
                   drift away in power from the median baseline powers. This
                   assumes the calibrator is compact, so all baselines should
                   see the same amplitude.
                4. Baselines (per channel, correlation) that have a
                   significant portion of their calibrator amplitudes flagged
                5. Baselines (per channel, correlation) that have a
                   significant portion of their calibrator phases flagged

        :param kwargs:
            "msname" : name of measurement set
            "data_column" : data column to use for amplitude flagging
            "field" : Comma-seperated list of fields
            "cal_field" : Comma-seperated list of calibrator fields
            "nrows_chunk" : number of rows to read per chunk
            "simulate" :   if true then the only statistics are calculated
            "nthreads" :   limits the number of threads to run on, default: all
            "phase_range_clip" :  Valid calibrator phase range (in degrees)
                                  specified in the CASA range format
                                  'float~float'
            "amp_frac_clip" : flag baseline channel when scan mean amplitude
                              exceeds themedian power over all scans by
                              (1 +/- amp_frac_clip)
            "invalid_count_frac_clip" : Maximum number of data points (per
                                        correlation per channel per baseline
                                        over all calibrator scans per baseline)
                                        to be invalid before a baseline
                                        is deemed untrustworthy (as fraction of
                                        unflagged data for that baseline channel)
            "output_dir"  : Name of output dir where to dump diagnostic plots


        :post conditions:
            if simulate is false the measurement flags will be modified
        """
        ms_meta = antenna_tasks.check_ms(**kwargs)
        ms = str(kwargs["msname"])

        try:
            DATA = str(kwargs["data_column"])
        except:
            raise ValueError("flag_excessive_amp_phase_error expects a data column (key 'data_column') as input")

        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["field"]):
                raise ValueError("Expect list of field identifiers")
            fields = [int(f) for f in kwargs["field"].split(",")]
        except:
            raise ValueError("flag_excessive_amp_phase_error expects a field(s) (key "
                             "'field') as input")

        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["cal_field"]):
                raise ValueError("Expect list of field identifiers")
            cal_fields = [int(f) for f in kwargs["cal_field"].split(",")]
        except:
            raise ValueError("flag_excessive_amp_phase_error expects calibrator field(s) (key "
                             "'cal_field') as input")

        if not set(cal_fields).issubset(set(fields)):
            raise ValueError("Calibrator fields must be subset of fields "
                             "that must be flagged.")
        try:
            max_amp_frac_clip = float(kwargs["amp_frac_clip"])
        except:
            raise ValueError("flag_excessive_amp_phase_error expects a maximum number "
                             "of sigmas for flagging drifts"
                             "(key 'amp_frac_clip') as input")

        try:
            nrows_to_read = int(kwargs["nrows_chunk"])
        except:
            raise ValueError("flag_excessive_amp_phase_error expects number of rows to read per chunk "
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
            raise ValueError("flag_excessive_amp_phase_error expects simulate flag "
                             "(key 'simulate') as input")

        try:
            max_inv_vis = float(kwargs["invalid_count_frac_clip"])
        except:
            raise ValueError("flag_excessive_amp_phase_error expects fraction of maximum invalid"
                             "visibilities before flagging entire baselines"
                             "from the observation (key 'invalid_count_frac_clip') "
                             "as input")

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
            raise ValueError("flag_excessive_amp_phase_error expects a valid phase range "
                             "(key 'phase_range_clip') as input "
                             "with format 'float~float' in degrees.")

        try:
            nthreads = int(kwargs["nthreads"])
        except:
            nthreads=multiprocessing.cpu_count()
            politsiyakat.log.warn("nthreads not specified. I will use %d "
                                  "threads" % nthreads)
        try:
            output_dir = str(kwargs["output_dir"])
        except:
            raise ValueError("flag_excessive_amp_phase_error expects an output_directory "
                             "(key 'output_dir') as input")


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

        executor = ThreadPoolExecutor(max_workers=nthreads)

        source_scan_info = {}
        obs_start = None
        for chunk_i in xrange(nchunk):
            politsiyakat.log.info("Processing chunk %d of %d..." %
                (chunk_i + 1, nchunk))
            politsiyakat.log.info("\tReading MS")
            kwargs["chunk_id"] = chunk_i
            kwargs["ack"] = False
            maintable_chunk = antenna_tasks.read_ms_maintable_chunk(**kwargs)
            a1 = maintable_chunk["a1"]
            a2 = maintable_chunk["a2"]
            baseline = maintable_chunk["baseline"]
            field = maintable_chunk["field"]
            data = maintable_chunk["data"]
            flag = maintable_chunk["flag"]
            desc = maintable_chunk["desc"]
            spw = maintable_chunk["spw"]
            scan = maintable_chunk["scan"]
            time = maintable_chunk["time"]
            obs_start = min(obs_start, np.min(time)) \
                if obs_start is not None else np.min(time)

            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tProcessing field %s (total %d fields)"
                                      % (source_names[field_i], no_fields))
                if field_id not in source_scan_info:
                    source_scan_info[field_id] = {
                        "is_calibrator": (field_id in cal_fields),
                        "scan_list": [],
                    }
                source_rows = np.argwhere(field == field_id)
                source_scans = np.unique(scan[source_rows])
                if source_scans.size == 0:
                    politsiyakat.log.info("\t\tNo source scans found in this chunk.")
                    continue
                politsiyakat.log.info("\t\tFound %d scans for this field in"
                                      " the chunk:"
                                      % source_scans.size)
                for s in source_scans:
                    scan_rows = np.argwhere(scan == s)
                    scan_start = np.min(time[scan_rows])
                    scan_end = np.max(time[scan_rows])
                    if s not in source_scan_info[field_id]:
                        source_scan_info[field_id][s] = {
                            "scan_start" : scan_start,
                            "scan_end" : scan_end,
                            "mean_power" : np.zeros([no_baselines,
                                                     nchan * nspw,
                                                     ncorr]),
                            "num_unflagged_vis" : np.zeros([no_baselines,
                                                            nchan * nspw,
                                                            ncorr]),
                            "num_phase_error" : np.zeros([no_baselines,
                                                          nchan * nspw,
                                                          ncorr]),
                            "num_chunks" : 1,
                        }
                        source_scan_info[field_id]["scan_list"].append(s)
                        politsiyakat.log.info("\t\t\tChunk contains start of a new"
                                              " scan %d, duration %ds" %
                                              (s, scan_end - scan_start))
                    else:
                        source_scan_info[field_id][s]["num_chunks"] += 1
                        source_scan_info[field_id][s]["scan_start"] = \
                            min(source_scan_info[field_id][s]["scan_start"], scan_start)
                        source_scan_info[field_id][s]["scan_end"] = \
                            max(source_scan_info[field_id][s]["scan_end"], scan_end)
                        politsiyakat.log.info("\t\t\tUpdating start and end times"
                                              " for scan %d, duration %ds" %
                                              (s,
                                               source_scan_info[field_id][s]["scan_end"]
                                               - source_scan_info[field_id][s]["scan_start"]))
                    notflag = np.logical_not(flag)
                    for spw_i in xrange(nspw):
                        futures_list = []
                        data_in_scan_spw = \
                            np.logical_and(scan == s,
                                           spw == spw_i)

                        for bi in xrange(no_baselines):
                            def compute_bl_stats(bi,
                                                 flag,
                                                 data,
                                                 data_in_scan_spw,
                                                 ncorr,
                                                 nchan,
                                                 is_cal_field):
                                # (nrows, nchan, ncorr)
                                data_in_bi_scan_spw = np.argwhere(
                                    np.logical_and(
                                        data_in_scan_spw,
                                        baseline == bi))

                                # Now select all unflagged data of this
                                # baseline, within this field and spw
                                selection_data = data[data_in_bi_scan_spw]
                                selection_nf = notflag[data_in_bi_scan_spw]
                                selection_flagged = selection_data * \
                                    selection_nf
                                amp_selection = np.abs(selection_flagged)

                                # (chan, corr) for each baseline of a field
                                count_unflagged = np.sum(selection_nf,
                                                         axis=0)

                                # compute mean of this chunk of
                                # field data for this baseline
                                sum_unflagged = \
                                    np.sum(amp_selection,
                                           axis=0,
                                           dtype=np.float64)

                                # if the scan is that of a calibrator
                                # clip the phase if it is out of range
                                if is_cal_field:
                                    selection_f = flag[data_in_bi_scan_spw]
                                    count_flagged = np.sum(selection_f,
                                                           axis=0)
                                    phase_selection = \
                                        np.angle(selection_flagged)
                                    flag[data_in_bi_scan_spw] = np.logical_or(
                                        flag[data_in_bi_scan_spw],
                                        np.logical_or(
                                            phase_selection <
                                                low_valid_phase,
                                            phase_selection >
                                                high_valid_phase))
                                    pf = flag.view()
                                    count_pf = np.sum(flag[data_in_bi_scan_spw],
                                                      axis=0) - count_flagged
                                else:
                                    pf = flag.view()
                                    count_pf = np.zeros([nchan, ncorr])

                                return (count_unflagged.reshape([nchan, ncorr]),
                                        sum_unflagged.reshape([nchan, ncorr]),
                                        pf,
                                        count_pf.reshape([nchan, ncorr]))

                            futures_list.append(
                                executor.submit(compute_bl_stats,
                                                bi,
                                                flag.copy(),
                                                data.view(),
                                                data_in_scan_spw.view(),
                                                ncorr,
                                                nchan,
                                                source_scan_info[field_id]["is_calibrator"]))

                        for bi in xrange(no_baselines):
                            bl_unflagged_vis_count,\
                            bl_sum_unflagged,\
                            pf,\
                            bl_count_pf = futures_list[bi].result()
                            # just store the running sum for now
                            source_scan_info[field_id][s]["mean_power"] \
                                 [bi,(spw_i*nchan):((spw_i+1)*nchan),
                                 :] += bl_sum_unflagged
                            # along with counters
                            source_scan_info[field_id][s]["num_unflagged_vis"]\
                                 [bi,
                                  (spw_i*nchan):((spw_i+1)*nchan),
                                  :] += bl_unflagged_vis_count
                            source_scan_info[field_id][s]["num_phase_error"]\
                                 [bi,
                                  (spw_i*nchan):((spw_i+1)*nchan),
                                  :] += bl_count_pf
                            if not simulate:
                                flag = np.logical_or(pf, flag) #reduce

            # Write flags of the calibrators back to the ms
            if not simulate:
                politsiyakat.log.info("\tWriting phase clip flags "
                                      "of calibrators to MS")
                with table(ms, readonly=False, ack=False) as t:
                    t.putcol("FLAG",
                             flag,
                             chunk_i * nrows_to_read,
                             min(t.nrows() - (chunk_i * nrows_to_read),
                                 nrows_to_read))

        politsiyakat.log.info("Looking for baselines with general phase error "
                              "across all calibrator scans...")

        # histogram the baselines per channel and correlation
        histogram_phase_off = np.zeros([no_baselines,
                                        nchan * nspw,
                                        ncorr])
        histogram_total_unflagged = np.zeros([no_baselines,
                                              nchan * nspw,
                                              ncorr])
        for field_id in cal_fields:
            scan_list = source_scan_info[field_id]["scan_list"]
            for s in scan_list:
                histogram_phase_off += \
                    source_scan_info[field_id][s]["num_phase_error"]
                histogram_total_unflagged += \
                    source_scan_info[field_id][s]["num_unflagged_vis"]

        # Clip baselines (per channel and correlation) that were 
        # dominated by phase error during the calibrator scans
        blnz = (histogram_total_unflagged != 0)
        histogram_percentage_phase_off = \
            (histogram_phase_off /
             (histogram_total_unflagged + 0.00000000001)) * blnz
        clip_baselines_phases = \
            histogram_percentage_phase_off > max_inv_vis
        # (nbl, ncorr)
        histogram_percentage_chan_bl_phase_off = np.sum(clip_baselines_phases,
                                                        axis=1)
        flagged_bls_on_phase = False
        for bi in np.argwhere(np.sum(histogram_percentage_chan_bl_phase_off,
                                     axis=1) > 0):
            politsiyakat.log.info("\tBaseline %d has %s untrustworthy "
                                  "channels per correlation that was not previously "
                                  "flagged." %
                (bi,
                 ",".join([str(cnt) for cnt in
                           histogram_percentage_chan_bl_phase_off[bi]])))
            flagged_bls_on_phase = True
        if not flagged_bls_on_phase:
            politsiyakat.log.info("\tDid not find any baselines with significant "
                                  "overall phase drifts when comparing "
                                  "across all calibrator scans")

        # Clip baselines per channel and correlation in scans that
        # are well above the median over all scans
        for field_i, field_id in enumerate(fields):
            politsiyakat.log.info("Doing interscan comparisons for field "
                                  "%s (total %d fields)" %
                                  (source_names[field_i], no_fields))
            scan_list = source_scan_info[field_id]["scan_list"]

            # Compute mean power per baseline channel per scan
            # then compute the median of these scan baseline channel
            # means
            list_bl_chan_powers = []
            for s in scan_list:
                #(nbl, nchan, ncorr)
                bl_chan_mean = source_scan_info[field_id][s]["mean_power"] /\
                    (source_scan_info[field_id][s]["num_unflagged_vis"] +
                     0.000000001)
                bl_chan_mean *= (source_scan_info
                                    [field_id][s]["num_unflagged_vis"] != 0)
                source_scan_info[field_id][s]["mean_power"] = bl_chan_mean
                list_bl_chan_powers.append(
                    source_scan_info[field_id][s]["mean_power"])

                # Print out some stats for this scan
                politsiyakat.log.info("\tScan %d has mean powers:"
                                      %s)
                for ci in xrange(ncorr):
                    data_corr = source_scan_info\
                        [field_id][s]["mean_power"][:,:,ci]
                    count_corr = np.sum(source_scan_info
                        [field_id][s]["num_unflagged_vis"][:,:,ci] > 0)
                    S = np.sum(data_corr)
                    C = np.sum(count_corr)
                    mean_pow = (S / C) if C != 0 else 0
                    politsiyakat.log.info("\t\tCorr %d: %f" %
                        (ci, mean_pow))

                politsiyakat.log.info("\tNumber of clipped phases "
                                      "of scan %d:" % s)

                for ci in xrange(ncorr):
                    clipped_vis_count = \
                        source_scan_info\
                            [field_id][s]["num_phase_error"][:,:,ci]
                    count_corr = np.sum(source_scan_info
                         [field_id][s]["num_unflagged_vis"][:,:,ci])
                    S = np.sum(clipped_vis_count)
                    C = np.sum(count_corr)
                    P = (S / C * 100) if C != 0 else 0
                    politsiyakat.log.info("\t\tCorr %d: %d (%.2f%%)" %
                        (ci, int(S), P))

            # (nbl, nchan, ncorr) median per baseline, channel between scans
            median_scan_power = np.median(np.array(list_bl_chan_powers),
                                          axis=0)

            # (nchan, ncorr), median per baseline channel
            median_chan_power = np.median(median_scan_power,
                                          axis=0)

            # the per baseline comparison between scans are safe to use
            # for both calibrator and target fields
            for s in scan_list:
                politsiyakat.log.info("\tComparing baseline amplitudes between "
                                      "scan %d and other scans for field %s (total "
                                      "%d scans)" %
                                      (s,
                                       source_names[field_i],
                                       len(scan_list)))
                hot_bl_scan = \
                    (source_scan_info[field_id][s]["mean_power"] >
                     (1 + max_amp_frac_clip) * median_scan_power)
                cold_bl_scan = \
                    (source_scan_info[field_id][s]["mean_power"] <
                     (1 - max_amp_frac_clip) * median_scan_power)
                source_scan_info[field_id][s]["hot_bl_scan"] = \
                    hot_bl_scan
                source_scan_info[field_id][s]["cold_bl_scan"] = \
                    cold_bl_scan
                hot_cold_found = False
                hist_hot_bl_scan = np.sum(hot_bl_scan, axis=1)
                hist_cold_bl_scan = np.sum(cold_bl_scan, axis=1)
                for bi in np.argwhere(np.sum(hist_hot_bl_scan,
                                             axis=1) > 0):
                    politsiyakat.log.info("\t\tBaseline %d is running hot "
                                          "in scan %d in %s channels" %
                                          (bi, s,
                                           ",".join(str(cnt) for cnt in
                                                    hist_hot_bl_scan[bi])))
                    hot_cold_found = True
                for bi in np.argwhere(np.sum(hist_cold_bl_scan,
                                             axis=1) > 0):
                    politsiyakat.log.info("\t\tBaseline %d is running cold "
                                          "in scan %d in %s channels" %
                                          (bi, s,
                                           ",".join(str(cnt) for cnt in
                                                    hist_cold_bl_scan[bi])))
                    hot_cold_found = True

                if not hot_cold_found:
                    politsiyakat.log.info("\t\tNo baselines amplitudes are "
                                          "fluctuating comparing between "
                                          "scans")

            source_scan_info[field_id]["median_scan_power"] = \
                median_scan_power

            # when looking at corrected calibrator fields then we 
            # can compare amplitudes between baselines and flag
            # out overall hot and cold baselines.
            hot_bl = np.zeros([no_baselines,
                               nchan,
                               ncorr])
            cold_bl = np.zeros([no_baselines,
                                nchan,
                                ncorr])

            if field_id in cal_fields:
                politsiyakat.log.info("\tComparing baseline amplitudes across "
                                      "baseline in calibrator field %s" %
                                      source_names[fields.index(field_id)])
                hot_cold_found = False

                for bi in xrange(no_baselines):
                    hot_chans = (median_scan_power[bi,:,:] >
                        (1 + max_amp_frac_clip) * median_chan_power)
                    cold_chans = (median_scan_power[bi,:,:] <
                        (1 - max_amp_frac_clip) * median_chan_power)
                    hot_bl[bi, :, :] += hot_chans
                    cold_bl[bi, :, :] += cold_chans
                    if np.any(hot_chans):
                        politsiyakat.log.info("\t\tBaseline %d is running hot "
                                              "in %s channels when comparing "
                                              "across baselines" %
                                              (bi,
                                               ",".join(str(cnt) for cnt in
                                                        np.sum(hot_chans,
                                                               axis=0))))
                        hot_cold_found = True
                    if np.any(cold_chans):
                        politsiyakat.log.info("\t\tBaseline %d is running cold "
                                              "in %s channels when comparing "
                                              "across baselines" %
                                              (bi,
                                               ",".join(str(cnt) for cnt in
                                                        np.sum(cold_chans,
                                                               axis=0))))
                        hot_cold_found = True
                if not hot_cold_found:
                    politsiyakat.log.info("\t\tNo baselines found that had "
                                          "generally higher amplitudes than "
                                          "the median array amplitude.")
            source_scan_info[field_id]["hot_baseline_count"] = hot_bl
            source_scan_info[field_id]["cold_baseline_count"] = cold_bl

        # Now clip channels in baselines with significant amount of
        # data much hotter or colder than the array median
        politsiyakat.log.info("Looking for baselines with general amplitude error "
                              "across all calibrator scans...")
        histogram_amp_off = np.zeros([no_baselines,
                                      nchan * nspw,
                                      ncorr])
        for field_id in cal_fields:
            histogram_amp_off += \
                source_scan_info[field_id]["hot_baseline_count"] +\
                source_scan_info[field_id]["cold_baseline_count"]

        blnz = (histogram_total_unflagged != 0)
        histogram_percentage_amp_off = \
            (histogram_amp_off /
             (histogram_total_unflagged + 0.00000000001)) * blnz
        clip_baselines_amps = \
            histogram_percentage_amp_off > max_inv_vis

        # (nbl, ncorr)
        histogram_percentage_chan_bl_amp_off = np.sum(clip_baselines_phases,
                                                      axis=1)
        flagged_bls_on_amp = False
        for bi in np.argwhere(np.sum(histogram_percentage_chan_bl_amp_off,
                                     axis=1) > 0):
            politsiyakat.log.info("\tBaseline %d has %s untrustworthy "
                                  "channels per correlation that was not previously "
                                  "flagged." %
                (bi,
                 ",".join([str(cnt) for cnt in
                           histogram_percentage_chan_bl_amp_off[bi]])))
            flagged_bls_on_amp = True

        clip_baseline_phases_amps = np.logical_or(clip_baselines_phases,
                                                  clip_baselines_amps)
        for chunk_i in xrange(nchunk):
            politsiyakat.log.info("Applying flags in chunk %d of %d..." %
                (chunk_i + 1, nchunk))
            politsiyakat.log.info("\tReading MS")
            kwargs["chunk_id"] = chunk_i
            kwargs["ack"] = False
            kwargs["read_exclude"] = ["data", "time"]
            maintable_chunk = antenna_tasks.read_ms_maintable_chunk(**kwargs)
            a1 = maintable_chunk["a1"]
            a2 = maintable_chunk["a2"]
            baseline = maintable_chunk["baseline"]
            field = maintable_chunk["field"]
            flag = maintable_chunk["flag"]
            desc = maintable_chunk["desc"]
            spw = maintable_chunk["spw"]
            scan = maintable_chunk["scan"]

            for field_i, field_id in enumerate(fields):
                # Clip baselines accross all fields
                # that had general phase and amplitude
                # errors accross all calibrator scans
                flagged_baselines = np.argwhere(clip_baseline_phases_amps)
                for spw_i in xrange(nspw):
                    for (bi, ch_i, corr_i) in flagged_baselines:
                        apply_rows = \
                            np.argwhere(np.logical_and(baseline == bi,
                                                       spw == spw_i))
                        flag[apply_rows, ch_i % nchan, corr_i] = True

                # if the source is a calibrator we can also flag out
                # times where individual baseline channels drifted 
                # from the median power for that channel accross
                # all scans
                if source_scan_info[field_id]["is_calibrator"]:
                    flagged_baselines = np.argwhere(np.logical_or(
                        source_scan_info[field_id]["hot_baseline_count"] > 0,
                        source_scan_info[field_id]["cold_baseline_count"] > 0))
                    for spw_i in xrange(nspw):
                        for (bi, ch_i, corr_i) in flagged_baselines:
                            apply_rows = \
                                np.argwhere(np.logical_and(baseline == bi,
                                                           spw == spw_i))
                            flag[apply_rows, ch_i % nchan, corr_i] = True

                # The scan-based phase clips were already applied to the
                # calibrators during power calculations, all that remains 
                # is to apply scan-based amplitude flags per baseline, 
                # channel and correlation
                scan_list = source_scan_info[field_id]["scan_list"]
                for s in scan_list:
                    flagged_baselines = np.argwhere(np.logical_or(
                        source_scan_info[field_id][s]["hot_bl_scan"] > 0,
                        source_scan_info[field_id][s]["cold_bl_scan"] > 0))
                    for spw_i in xrange(nspw):
                        for (bi, ch_i, corr_i) in flagged_baselines:
                            apply_rows = \
                                np.argwhere(np.logical_and(np.logical_and(
                                    baseline == bi,
                                    spw == spw_i),
                                    scan == s))
                            flag[apply_rows, ch_i % nchan, corr_i] = True

            # Write flags of the fields and scans back to the ms
            if not simulate:
                politsiyakat.log.info("\t\tWriting field to MS")
                with table(ms, readonly=False, ack=False) as t:
                    t.putcol("FLAG",
                             flag,
                             chunk_i * nrows_to_read,
                             min(t.nrows() - (chunk_i * nrows_to_read),
                                 nrows_to_read))

        uv_dist = uv_dist_per_baseline(no_baselines,
                                       nant,
                                       antenna_positions)

        # Dump a diagnostic plot of the number of bad phase channels per
        # baseline
        for c in xrange(ncorr):
            fig = plt.figure()
            ranked_uv_dist = np.argsort(uv_dist)
            plt.plot(uv_dist[ranked_uv_dist],
                     histogram_percentage_chan_bl_phase_off[ranked_uv_dist, c])
            plt.title(("Flag excessive phase error (corr %d) " % c) + os.path.basename(ms))
            plt.xlabel("UVdist (m)")
            plt.ylabel("Number of bad previously unflagged channels")
            fig.savefig(output_dir + "/%s-FLAGGED_PHASE_UVDIST.OBSWIDE.CORR_%d.png" %
                        (os.path.basename(ms),
                         c))

            plt.close(fig)

        # Dump a diagnostic plot of the number of bad amplitude channels per
        # baseline
        for c in xrange(ncorr):
            fig = plt.figure()
            ranked_uv_dist = np.argsort(uv_dist)
            plt.plot(uv_dist[ranked_uv_dist],
                     histogram_percentage_chan_bl_amp_off[ranked_uv_dist, c])
            plt.title(("Flag excessive amplitude error (corr %d) " % c) + os.path.basename(ms))
            plt.xlabel("UVdist (m)")
            plt.ylabel("Number of bad previously unflagged channels")
            fig.savefig(output_dir + "/%s-FLAGGED_AMP_UVDIST.OBSWIDE.CORR_%d.png" %
                        (os.path.basename(ms),
                         c))
            plt.close(fig)

        # Dump diagnostic plots of the number of amplitude and phase clips applied per
        # calibrator field
        for field_i, field_id in enumerate(fields):
            if not source_scan_info[field_id]["is_calibrator"]:
                continue
            hist_hot = np.sum(source_scan_info
                                [field_id]["hot_baseline_count"] > 0,
                              axis=1)
            hist_cold = np.sum(source_scan_info
                                [field_id]["cold_baseline_count"] > 0,
                               axis=1)
            for c in xrange(ncorr):
                fig = plt.figure()
                ranked_uv_dist = np.argsort(uv_dist)
                plt.plot(uv_dist[ranked_uv_dist],
                         hist_cold[ranked_uv_dist,c],
                         "b", label="#Cold channels")
                plt.plot(np.sqrt(uv_dist[ranked_uv_dist]),
                         hist_hot[ranked_uv_dist,c],
                         "r", label="#Hot channels")
                plt.title(("Clip baseline amp offset (corr %d, field"
                          " %s) " % (c, source_names[field_i])) + os.path.basename(ms))
                plt.xlabel("UVdist (m)")
                plt.ylabel("Number of bad previously unflagged channels")
                plt.legend()
                fig.savefig(output_dir + "/%s-FLAGGED_AMP_UVDIST.CALFIELD_%s.CORR_%d.png" %
                            (os.path.basename(ms),
                             field_id,
                             c))
                plt.close(fig)
            total_phase_clip = np.zeros([no_baselines,
                                         nchan,
                                         ncorr])
            scan_list = source_scan_info[field_id]["scan_list"]
            for s in scan_list:
                # (nbl, nchan, ncorr)
                total_phase_clip += source_scan_info[field_id][s]["num_phase_error"]
            # (nchan, ncorr)
            hist_phase_clip = np.sum(total_phase_clip > 0,
                                     axis=1)
            for c in xrange(ncorr):
                fig = plt.figure()
                ranked_uv_dist = np.argsort(uv_dist)
                plt.plot(uv_dist[ranked_uv_dist],
                         hist_phase_clip[ranked_uv_dist,c])
                plt.title(("Clip calibrator channels due to phase (corr %d, field"
                          " %s) " % (c, source_names[field_i])) + os.path.basename(ms))
                plt.xlabel("UVdist (m)")
                plt.ylabel("Number of bad previously unflagged channels")
                fig.savefig(output_dir + "/%s-FLAGGED_PHASE_UVDIST.CALFIELD_%s.CORR_%d.png" %
                            (os.path.basename(ms),
                             field_id,
                             c))
                plt.close(fig)

        # Dump diagnostic plots of the amplitude clips done per scan
        # (intra-baseline basis)
        for field_i, field_id in enumerate(fields):
            scan_list = source_scan_info[field_id]["scan_list"]
            for c in xrange(ncorr):
                fig = plt.figure()
                cmcool = plt.get_cmap("cool")
                for s in scan_list:
                    s_start = source_scan_info\
                                        [field_id][s]["scan_start"]
                    s_end = source_scan_info\
                                        [field_id][s]["scan_end"]
                    s_mid = s_start + (s_end - s_end) * 0.5 - obs_start
                    hist_amp = np.sum((source_scan_info
                                        [field_id][s]["hot_bl_scan"] +
                                      source_scan_info
                                        [field_id][s]["cold_bl_scan"]) > 0,
                                      axis=1)
                    for bl in xrange(no_baselines):
                        plt.scatter(np.repeat(s_mid, no_baselines),
                                    hist_amp[:, c],
                                    c=np.repeat(uv_dist[bl], no_baselines),
                                    vmin=np.min(uv_dist),
                                    vmax=np.max(uv_dist),
                                    cmap=cmcool,
                                    s=0.1,
                                    marker=".")
                plt.colorbar()
                plt.title(("Flagged amp time fluctuations (col:uvdist, "
                           "corr %d, field %s) " % (c, source_names[field_i])) +
                          os.path.basename(ms))
                plt.xlabel("Time")
                plt.ylabel("Number of bad previously unflagged channels")
                fig.savefig(output_dir +
                            "/%s-FLAGGED_AMP_SCAN.FIELD_%s.CORR_%d.png" %
                            (os.path.basename(ms),
                             field_id,
                             c))
                plt.close(fig)

        executor.shutdown()

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
        ms_meta = antenna_tasks.check_ms(**kwargs)
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
            maintable_chunk = antenna_tasks.read_ms_maintable_chunk(**kwargs)
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
    def flag_antennas_with_few_baselines_remaining(cls, **kwargs):
        """
            Flags antennae (ANTENNA table flags) that have too few
            baselines remaining - should be more than the number of antennae.
        :param kwargs:
            "msname" : name of measurement set
            "data_column" : name of data column
            "field" : field number(s) to inspect, comma-seperated
            "nrows_chunk" : Number of rows to read per chunk (reduces memory
                            footprint
            "simulate" : Only simulate and compute statistics, don't actually
                         flag anything.
            "min_bls_per_ant" : Minimum number of baselines per antenna
                                before flagging antenna out of solvable
                                antennae list. If not specified defaults to
                                nant + 1
        :post-conditions:
            Measurement set is reflagged to invalidate all baselines affected by
            severe phase error.
        """
        ms_meta = antenna_tasks.check_ms(**kwargs)
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
            raise ValueError("flag_excessive_delay_error expects field(s) (key "
                             "'field') as input")
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
        try:
            min_bls_per_ant = int(kwargs["min_bls_per_ant"])
        except:
            min_bls_per_ant = ms_meta["nant"] + 1
            politsiyakat.log.warn("Setting minimum number of baselines to nant"
                                  " + 1")

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
        politsiyakat.log.info("Inspecting the following fields:")
        for fi, f in enumerate(fields):
            politsiyakat.log.info("\t(%d): %s" % (f, source_names[fi]))
        bl_count_per_ant = np.zeros([len(fields),
                                     nant,
                                     no_baselines])
        for chunk_i in xrange(nchunk):
            politsiyakat.log.info("Processing chunk %d / %d" %
                                  (chunk_i + 1, nchunk))
            politsiyakat.log.info("\tReading MS")
            kwargs["chunk_id"] = chunk_i
            kwargs["ack"] = False
            kwargs["read_exclude"] = ["time", "data", "desc", "spw"]
            maintable_chunk = antenna_tasks.read_ms_maintable_chunk(**kwargs)
            a1 = maintable_chunk["a1"]
            a2 = maintable_chunk["a2"]
            field = maintable_chunk["field"]
            flag = maintable_chunk["flag"]
            baseline = maintable_chunk["baseline"]

            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tProcessing field %s (%d fields "
                                      "in total)" %
                                      (source_names[fields.index(field_id)],
                                       len(fields)))

                rows_with_field = (field == field_id)
                for ant in xrange(nant):
                    rows_with_ant_field = \
                        np.logical_and(rows_with_field,
                                       np.logical_or(a1 == ant,
                                                     a2 == ant))
                    #(nrow, nchan, ncorr)
                    unflagged_selection = \
                        np.logical_and(np.logical_not(flag),
                                       np.tile(rows_with_ant_field,
                                               (ncorr, nchan, 1)).T)
                    for row in np.argwhere(np.sum(np.sum(unflagged_selection,
                                                         axis=1),
                                                  axis=1) > 0):
                        bl_count_per_ant[field_i,
                                         ant,
                                         baseline[row]] += 1
        # count unique baselines per antenna
        # should be more than the minimum per field
        flag_ant = np.zeros([nant], dtype=np.bool)
        for field_i, field_id in enumerate(fields):
            flag_ant = \
                np.logical_or(flag_ant,
                              np.sum(bl_count_per_ant[field_i] > 0,
                                     axis=1) > min_bls_per_ant)

        politsiyakat.log.info("Looking for antennas with fewer than %d "
                              "unflagged baselines in one or more fields" %
                              min_bls_per_ant)
        any_flagged = False
        for fa in np.argwhere(flag_ant):
            politsiyakat.log.info("\tAntenna %s was mostly flagged. Will flag "
                                  "in ANTENNAS table" % antnames[fa])
            any_flagged = True
        if not any_flagged:
            politsiyakat.log.info("\tNo antennae were excessively flagged")

        # finally update the antenna table
        with table(ms + "::ANTENNA", readonly=False, ack=False) as t:
            orig_flag_ant = t.getcol("FLAG_ROW")
            updated_flag_ant = np.logical_or(orig_flag_ant,
                                             flag_ant)
            if not simulate:
                t.putcol("FLAG_ROW", updated_flag_ant)

