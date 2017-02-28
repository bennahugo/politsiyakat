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
    def flag_amplitude_drifts(cls, **kwargs):
        """
            Flags visibility amplitudes that fall several sigma away from the
            mean on a per field basis. Each channel and correlation is treated
            separately on a per baseline basis.
        :param kwargs:
            "ms_name" : name of measurement set
            "data_column" : data column to use for amplitude flagging
            "field" : Comma-seperated list of fields
            "cal_field" : Comma-seperated list of calibrator fields
            "nrow_chunk" : number of rows to read per chunk (should not be too
                           small, otherwise we won't be able to compute the
                           mean and std dev of the chunk reliably
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


        :post conditions:
            if simulate is false the measurement flags will be modified
        """
        ms_meta = antenna_tasks.check_ms(**kwargs)
        ms = str(kwargs["msname"])

        try:
            DATA = str(kwargs["data_column"])
        except:
            raise ValueError("flag_amplitude_drifts expects a data column (key 'data_column') as input")

        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["field"]):
                raise ValueError("Expect list of field identifiers")
            fields = [int(f) for f in kwargs["field"].split(",")]
        except:
            raise ValueError("flag_amplitude_drifts expects a field(s) (key "
                             "'field') as input")

        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["cal_field"]):
                raise ValueError("Expect list of field identifiers")
            cal_fields = [int(f) for f in kwargs["cal_field"].split(",")]
        except:
            raise ValueError("flag_amplitude_drifts expects calibrator field(s) (key "
                             "'cal_field') as input")

        try:
            max_amp_frac_clip = float(kwargs["amp_frac_clip"])
        except:
            raise ValueError("flag_amplitude_drifts expects a maximum number "
                             "of sigmas for flagging drifts"
                             "(key 'amp_frac_clip') as input")

        try:
            nrows_to_read = int(kwargs["nrows_chunk"])
        except:
            raise ValueError("flag_ampltude_drifts expects number of rows to read per chunk "
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
            raise ValueError("flag_amplitude_drifts expects simulate flag "
                             "(key 'simulate') as input")

        try:
            max_inv_vis = float(kwargs["invalid_count_frac_clip"])
        except:
            raise ValueError("flag_amplitude_drifts expects fraction of maximum invalid"
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
            raise ValueError("flag_amplitude_drifts expects a valid phase range "
                             "(key 'phase_range_clip') as input "
                             "with format 'float~float' in degrees.")

        try:
            nthreads = int(kwargs["nthreads"])
        except:
            nthreads=multiprocessing.cpu_count()
            politsiyakat.log.warn("nthreads not specified. I will use %d "
                                  "threads" % nthreads)

        source_names = [ms_meta["ms_field_names"][f] for f in fields]
        nchan = ms_meta["nchan"]
        nspw = ms_meta["nspw"]
        map_descriptor_to_spw = ms_meta["map_descriptor_to_spw"]
        nant = ms_meta["nant"]
        no_fields = len(fields)
        no_baselines = ms_meta["no_baselines"]
        ncorr = ms_meta["ncorr"]
        nchunk = ms_meta["nchunk"]

        politsiyakat.log.info("Will flag the following fields:")
        for fi, f in enumerate(fields):
            politsiyakat.log.info("\t(%d): %s" % (f, source_names[fi]) +
                                  " (calibrator)" if f in cal_fields else "")

        executor = ThreadPoolExecutor(max_workers=nthreads)

        source_scan_info = {}

        for chunk_i in xrange(12, nchunk):
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

            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tProcessing field %s (total %d fields)"
                                      % (source_names[field_i], no_fields))
                if field_id not in source_scan_info:
                    source_scan_info[field_id] = {
                        "is_calibrator": (field_id in cal_fields),
                    }
                source_rows = np.argwhere(field == field_id)
                source_scans = scan[source_rows]
                if len(source_rows) == 0:
                    politsiyakat.log.info("\t\tNo source scans found in this chunk.")
                    continue

                max_source_scan = np.max(source_scans)
                min_source_scan = np.min(source_scans)
                politsiyakat.log.info("\t\tFound %d scans for this field in"
                                      " the chunk:"
                                      % (max_source_scan -
                                         min_source_scan + 1))
                for s in xrange(min_source_scan, max_source_scan + 1):
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
                            "total_num_vis" : np.zeros([no_baselines,
                                                        nchan * nspw,
                                                        ncorr]),
                            "num_phase_error" : np.zeros([no_baselines,
                                                          nchan * nspw,
                                                          ncorr]),
                            "num_chunks" : 1,
                        }
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

                    for spw_i in xrange(nspw):
                        futures_list = []
                        data_in_scan_spw = \
                            np.tile(np.logical_and(scan == s,
                                                   spw == spw_i),
                                    (ncorr, nchan, 1)).T

                        for bi in xrange(no_baselines):
                            def compute_bl_stats(bi,
                                                 flag,
                                                 data,
                                                 data_in_scan_spw,
                                                 ncorr,
                                                 nchan,
                                                 is_cal_field):
                                # (nrows, nchan, ncorr)
                                data_in_bi_scan_spw = \
                                    np.logical_and(
                                        data_in_scan_spw,
                                        np.tile(baseline == bi,
                                                (ncorr,
                                                 nchan,
                                                 1)).T)
                                unflag_data = \
                                    np.logical_and(np.logical_not(flag),
                                                   data_in_bi_scan_spw)

                                # Now select all unflagged data of this
                                # baseline, within this field and spw
                                amp_selection = np.abs(data) * unflag_data

                                # (chan, corr) for each baseline of a field
                                count_unflagged = np.sum(
                                    unflag_data, axis=0)
                                count_total = np.sum(
                                    data_in_bi_scan_spw, axis=0)

                                # compute mean of this chunk of
                                # field data for this baseline
                                sum_unflagged = \
                                    np.sum(amp_selection,
                                           axis=0,
                                           dtype=np.float64)

                                # if the scan is that of a calibrator
                                # clip the phase if it is out of range
                                if is_cal_field:
                                    phase_selection = \
                                        np.angle(data) * unflag_data
                                    L = np.logical_and(
                                        data_in_bi_scan_spw,
                                        np.logical_or(
                                            phase_selection <
                                                low_valid_phase,
                                            phase_selection >
                                                high_valid_phase))
                                    pf = np.logical_or(flag,
                                                       L)
                                    count_pf = \
                                        np.count_nonzero(L, axis=0)
                                else:
                                    pf = flag.view()
                                    count_pf = np.zeros(nchan, ncorr)

                                return (count_unflagged,
                                        count_total,
                                        sum_unflagged,
                                        pf,
                                        count_pf)

                            futures_list.append(
                                executor.submit(compute_bl_stats,
                                                bi,
                                                flag.view(),
                                                data.view(),
                                                data_in_scan_spw.view(),
                                                ncorr,
                                                nchan,
                                                source_scan_info[field_id]["is_calibrator"]))

                        for bi in xrange(no_baselines):
                            bl_unflagged_vis_count,\
                            bl_total_vis_count,\
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
                            source_scan_info[field_id][s]["total_num_vis"]\
                                 [bi,
                                  (spw_i*nchan):((spw_i+1)*nchan),
                                  :] += bl_total_vis_count
                            source_scan_info[field_id][s]["num_phase_error"]\
                                 [bi,
                                  (spw_i*nchan):((spw_i+1)*nchan),
                                  :] += bl_count_pf
                            if not simulate:
                                flag = pf

                    # Write flagged of the calibrators back to the ms
                    if not simulate:
                        politsiyakat.log.info("\t\t\tWriting phase clip flags "
                                              "of calibrator to MS")
                        with table(ms, readonly=True, ack=False) as t:
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
            scan_list = [k for k in source_scan_info[field_id]
                         if isinstance(k, int)]
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
                                  "accross all calibrator scans")

        # Clip baselines per channel and correlation in scans that
        # are well above the median over all scans
        for field_id in fields:
            politsiyakat.log.info("Doing interscan comparisons for field "
                                  "%s (total %d fields)" %
                                  (source_names[field_i], no_fields))
            scan_list = [k for k in source_scan_info[field_id]
                         if isinstance(k, int)]

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
                                      "scans for this field (total "
                                      "%d scans)" %
                                      (len(scan_list)))
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
                                      "baseline in this calibrator field")
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
        for chunk_i in xrange(11, nchunk):
            politsiyakat.log.info("Processing chunk %d of %d..." %
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
                for (bi, ch_i, corr_i) in flagged_baselines:
                    bl_rows = np.argwhere(baseline == bi)
                    flag[bl_rows][ch_i, corr_i] = True

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
            "DATA" : name of data column
            "cal_field" : calibrator field number (preferably the bandpass calibrator)
            "valid_phase_range" : Phase range (in degrees) specified in the CASA range format
                                  'float~float'
            "max_invalid_datapoints" : Maximum number of data points (all
                                       correlations over all time per baseline
                                       channel) to be invalid before a baseline
                                       is deemed untrustworthy (as % of
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
            max_times = float(kwargs["max_invalid_datapoints"])
        except:
            raise ValueError("flag_excessive_delay_error expects a maximum invalid timesteps (\%) "
                             "(key 'max_invalid_datapoints') as input")

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
            antenna_positions = t.getcol("POSITION")
            nant = t.nrows()

        # be conservative autocorrelations is probably still in the mix
        # since they can be quite critical in calibration
        no_baselines = (nant * (nant - 1)) // 2 + nant

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

        with table(ms + "::POLARIZATION", readonly=True, ack=False) as t:
            ncorr = t.getcol("NUM_CORR")
        assert np.alltrue([ncorr[0] == ncorr[c] for c in xrange(len(ncorr))]), \
            "for now we can only handle rows that all have the same number correlations"
        ncorr = ncorr[0]

        # Lets keep a histogram of each channel (all unflagged data)
        # and a corresponding histogram channels where phase is very wrong
        histogram_data = np.zeros([no_baselines, nchan * nspw, ncorr],
                                  dtype=np.float64)
        histogram_phase_off = np.zeros([no_baselines, nchan * nspw, ncorr],
                                       dtype=np.float64)

        with table(ms, readonly=False, ack=False) as t:
            politsiyakat.log.info("Successfull read-write open of '%s'" % ms)
            nchunk = int(np.ceil(t.nrows() / float(nrows_to_read)))

            for chunk_i in xrange(nchunk):
                politsiyakat.log.info("Computing histogram for chunk %d / %d" %
                                      (chunk_i + 1, nchunk))
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

                for spw_i in xrange(nspw):
                    unflagged_data = data * \
                                     np.float64(np.logical_not(flag)) * \
                                     np.tile(field == cal_field,
                                             (ncorr, nchan, 1)).T * \
                                     np.tile(spw == spw_i, (ncorr, nchan, 1)).T
                    # Count all the places where there are unflagged correlations
                    S = (unflagged_data != 0.0)
                    # (nrows, nchan)
                    for r in xrange(len(S)):
                        histogram_data[baseline[r],
                                       (nchan*spw_i):(nchan * (spw_i + 1))] +=\
                            np.float64(S[r])
                    # Where there are some of the correlations outside
                    # valid phase range count, count them
                    ang = np.angle(unflagged_data)
                    less = ang < np.deg2rad(low_valid_phase)
                    more = ang > np.deg2rad(high_valid_phase)
                    L = np.logical_and((np.logical_or(less, more) >
                                        0), S)
                    # (nrows, nchan, ncorr)
                    for r in xrange(len(S)):
                        histogram_phase_off[baseline[r],
                                            (nchan*spw_i):(nchan * (spw_i + 1))] \
                            += np.float64(L[r])
            F = np.abs(histogram_phase_off / (histogram_data + 0.000000001)) > (max_times / 100.0)
            F *= (histogram_data != 0)
            no_channels_flagged_per_baseline = np.sum(F, axis=1)
            flagged_baseline_channels = np.argwhere(F)

            for bl_i, bl_sum in enumerate(no_channels_flagged_per_baseline):
                politsiyakat.log.info("Baseline %d has %s untrustworthy "
                                      "channels per correlation that was not previously "
                                      "flagged." % (bl_i, ",".join([str(cnt) for cnt in bl_sum])))
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
                for bl, chan, corr in flagged_baseline_channels:
                    flag[np.argwhere(baseline == bl), chan % nchan, corr] = True

                # finally actually touch the measurement set
                if not simulate:
                    t.putcol("FLAG",
                             flag,
                             chunk_i * nrows_to_read,
                             min(t.nrows() - (chunk_i * nrows_to_read),
                                 nrows_to_read))

            # Dump a diagnostic plot of the number of bad phase channels per
            # baseline
            for c in xrange(ncorr):
                fig = plt.figure()
                ranked_uvdist_sq = np.argsort(uv_dist_sq)
                plt.plot(np.sqrt(uv_dist_sq[ranked_uvdist_sq]),
                         no_channels_flagged_per_baseline[ranked_uvdist_sq, c])
                plt.title(("Flag excessive phase error (corr %d) " % c) + os.path.basename(ms))
                plt.xlabel("UVdist (m)")
                plt.ylabel("Number of bad previously unflagged channels")
                fig.savefig(output_dir + "%s-FLAGGED_PHASE_UVDIST.FIELD_%d.CORR_%d.png" %
                            (os.path.basename(ms), cal_field, c))
                plt.close(fig)

