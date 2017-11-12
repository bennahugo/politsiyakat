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
import politsiyakat
import re
from matplotlib import pyplot as plt
import scipy.interpolate as interp
from politsiyakat.data.misc import *
from politsiyakat.data.data_provider import data_provider

class flag_tasks:
    """
       Tasks Helper class

       Contains a number of tasks to detect and
       remove / fix cases where antennas have gone rogue.
    """

    def __init__(self):
        pass

    @classmethod
    def flag_phase_drifts(cls, **kwargs):
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
            severe amplitude error.
        """
        ms_meta = data_provider.check_ms(**kwargs)
        ms = str(kwargs["msname"])

        try:
            DATA = str(kwargs["data_column"])
        except:
            raise ValueError("flag_phase_drifts expects a data column (key 'data_column') as input")
        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["field"]):
                raise ValueError("Expect list of field identifiers")
            fields = [int(f) for f in kwargs["field"].split(",")]
        except:
            raise ValueError("flag_phase_drifts expects field(s) (key "
                             "'field') as input")

        try:
            if not re.match(r"^[0-9]+(?:,[0-9]+)*$", kwargs["cal_field"]):
                raise ValueError("Expect list of field identifiers")
            cal_fields = [int(f) for f in kwargs["cal_field"].split(",")]
        except:
            raise ValueError("flag_phase_drifts expects calibrator field(s) (key "
                             "'cal_field') as input")
        try:
            nrows_to_read = int(kwargs["nrows_chunk"])
        except:
            raise ValueError("flag_phase_drifts expects number of rows to read per chunk "
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
            raise ValueError("flag_phase_drifts expects simulate flag "
                             "(key 'simulate') as input")

        try:
            output_dir = str(kwargs["output_dir"])
        except:
            raise ValueError("flag_phase_drifts expects an output_directory "
                             "(key 'output_dir') as input")

        try:
            s2s_sigma = float(kwargs["scan_to_scan_threshold"])
        except:
            raise ValueError("flag_phase_drifts expect a scan to scan variation threshold (sigma) "
                             "(key 'scan_to_scan_threshold') as input")

        try:
            b2g_sigma = float(kwargs["baseline_to_group_threshold"])
        except:
            raise ValueError("flag_phase_drifts expect a sigma threshold for antenna variation from group"
                             "(key 'baseline_to_group_threshold') as input")

        try:
            dpi = int(kwargs["plot_dpi"])
        except:
            dpi = 300
            politsiyakat.log.warn("Warning: defaulting to plot dpi of 300. Keyword 'plot_dpi' "
                                  "can be used to control this behaviour.")

        try:
            plt_size = float(kwargs["plot_size"])
        except:
            plt_size = 6
            politsiyakat.log.warn("Warning: defaulting to plot size of 6 units. Keyword 'plot_size' can "
                                  "be used to control this behaviour.")
        if len(cal_fields) == 0:
            raise ValueError("Task 'flag_phase_drifts' requires at least one field marked as calibrator.")

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
                politsiyakat.log.info("Processing targets and calibrators in chunk %d of %d..." %
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
                        politsiyakat.log.info("\t\t\tField %s is not present in this chunk" % source_names[field_i])
                        continue

                    for s in source_scans:
                        scan_rows = np.argwhere(data["scan"] == s)
                        scan_start = np.min(data["time"][scan_rows])
                        scan_end = np.max(data["time"][scan_rows])
                        if s not in source_scan_info[field_id]:
                            source_scan_info[field_id][s] = {
                                "scan_start": scan_start,
                                "scan_end": scan_end,
                                "chisq_phase": np.zeros([no_baselines,
                                                        nchan * nspw,
                                                        ncorr]),
                                "chisq_amp": np.zeros([no_baselines,
                                                         nchan * nspw,
                                                         ncorr]),
                                "sum_phase": np.zeros([no_baselines,
                                                       nchan * nspw,
                                                       ncorr]),
                                "count_phase": np.zeros([no_baselines,
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
                            epic_name = 'antenna stats for %s' % source_names[field_i]
                            for a in xrange(nant):
                                politsiyakat.pool.submit_to_epic(epic_name,
                                                                 _wk_per_ant_flag_stats,
                                                                 a,
                                                                 ncorr,
                                                                 nchan,
                                                                 spw,
                                                                 field_id,
                                                                 s,
                                                                 data)
                            for bl in xrange(no_baselines):
                                politsiyakat.pool.submit_to_epic(epic_name,
                                                                 _wk_per_bl_phase_stats,
                                                                 bl,
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
                                    spw, totflagged, \
                                    totrowcount = r
                                    source_scan_info[field_id][s]["tot_flagged"][ant, spw * nchan:(spw + 1) * nchan, :] += \
                                        totflagged
                                    source_scan_info[field_id][s]["tot_rowcount"][ant, spw * nchan:(spw + 1) * nchan, :] += \
                                        totrowcount
                                elif r[0] == "blstat":
                                    task, bl, spw, blchisq, sph, cph = r
                                    source_scan_info[field_id][s]["chisq_phase"][bl, spw * nchan:(spw + 1) * nchan, :] += \
                                        blchisq
                                    source_scan_info[field_id][s]["sum_phase"][bl, spw * nchan:(spw + 1) * nchan, :] += \
                                        sph
                                    source_scan_info[field_id][s]["count_phase"][bl, spw * nchan:(spw + 1) * nchan, :] += \
                                        cph
                    tot_flagged = 0
                    tot_sel = 0
                    for s in source_scans:
                        tot_flagged += np.sum(source_scan_info[field_id][s]["tot_flagged"])
                        tot_sel += np.sum(source_scan_info[field_id][s]["tot_rowcount"])
                    if tot_sel != 0:
                        politsiyakat.log.info("\t\t\tField %s is %.2f %% flagged in this chunk" %
                                              (source_names[field_i],
                                               tot_flagged / tot_sel * 100.0))

            # Print some stats per field
            politsiyakat.log.info("Summary of flagging statistics per field:")
            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tField %s has the following scans:" % source_names[field_i])
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

            # flag based on median and std chisq per baseline channel
            flg_intra_bl = {}
            flg_inter_bl = {}
            for field_i, field_id in enumerate(fields):
                chisq_stacked = np.zeros([len(source_scan_info[field_id]["scan_list"]),
                                          no_baselines,
                                          nspw*nchan,
                                          ncorr])
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    chisq_stacked[si] = source_scan_info[field_id][s]["chisq_phase"]
                flgs_field_intrabl = np.zeros([len(source_scan_info[field_id]["scan_list"]),
                                              no_baselines, nchan * nspw, ncorr],
                                              dtype=np.bool)
                flgs_field_interbl = np.zeros([len(source_scan_info[field_id]["scan_list"]),
                                               no_baselines, nchan * nspw, ncorr],
                                              dtype=np.bool)
                mean_intrabl_phase = np.nanmean(chisq_stacked, axis=0)
                std_intrabl_phase = np.nanstd(chisq_stacked, axis=0)
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    flgs_field_intrabl[si, :, :, :] = np.abs(chisq_stacked[si] - mean_intrabl_phase) > \
                                                      s2s_sigma * std_intrabl_phase
                median_array_phase = np.nanmean(mean_intrabl_phase, axis=0)
                std_array_phase = np.nanstd(mean_intrabl_phase, axis=0)
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    flgs_field_interbl[si, :, :, :] = np.abs(chisq_stacked[si] - median_array_phase) > \
                                                      s2s_sigma * std_array_phase
                flg_intra_bl[field_id] = flgs_field_intrabl
                flg_inter_bl[field_id] = flgs_field_interbl

            # Print out new flagging statistics
            politsiyakat.log.info("The following calibrator flags resulted from intra-baseline flagging criterion:")
            for field_i, field_id in enumerate(fields):
                if field_id not in cal_fields:
                    continue
                politsiyakat.log.info("\tField %s:" % source_names[field_i])
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    flg = np.nansum(np.nansum(flg_intra_bl[field_id][si], axis=0),
                                    axis=0)
                    cnt = float(no_baselines * nchan)
                    politsiyakat.log.info("\t\tScan %s is [%s] %% flagged" %
                                          (s, ",".join([str(v) for v in flg / cnt * 100.0])))

            politsiyakat.log.info("The following calibrator flags resulted from inter-baseline flagging criterion:")
            for field_i, field_id in enumerate(fields):
                if field_id not in cal_fields:
                    continue
                politsiyakat.log.info("\tField %s:" % source_names[field_i])
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    flg = np.nansum(np.nansum(flg_inter_bl[field_id][si], axis=0),
                                    axis=0)
                    cnt = float(no_baselines * nchan)
                    politsiyakat.log.info("\t\tScan %s is [%s] %% flagged" %
                                          (s, ",".join([str(v) for v in flg / cnt * 100.0])))

            # compute scans onto which calibrator flags should be applied
            # assume all scans in-between calibrator scans are equally bad
            cal_scans = set()
            field_scans = set()
            for field_i, field_id in enumerate(fields):
                field_scans = field_scans.union(set(source_scan_info[field_id]["scan_list"]))
                if field_id not in cal_fields:
                    continue
                cal_scans = cal_scans.union(set(source_scan_info[field_id]["scan_list"]))
            cal_scans = sorted(list(cal_scans))
            field_scans = sorted(list(field_scans))
            apply_grps = {}
            for cs_i, cs in enumerate(cal_scans):
                c_grp_start = cal_scans[cs_i]
                c_grp_end = cal_scans[cs_i + 1] if cs_i < len(cal_scans) - 1 else np.inf
                apply_grps[cs] = [s for s in field_scans if (s >= c_grp_start and s < c_grp_end)]
                politsiyakat.log.info("Flags for calibrator scan %s will be applied to scans [%s]" %
                                      (cs, ",".join([str(s) for s in apply_grps[cs]])))
            new_flags = {}
            for sc in apply_grps.keys():
                for field_i, field_id in enumerate(fields):
                    if sc in source_scan_info[field_id]["scan_list"]:
                        sc_i = source_scan_info[field_id]["scan_list"].index(sc)
                        calflgs = np.logical_or(flg_intra_bl[field_id][sc_i],
                                                flg_inter_bl[field_id][sc_i])
                        for s in apply_grps[sc]:
                            for appfield_i, appfield_id in enumerate(fields):
                                if s in source_scan_info[appfield_id]["scan_list"]:
                                    if appfield_id in new_flags.keys():
                                        new_flags[appfield_id][s] = calflgs
                                    else:
                                        new_flags[appfield_id] = {s: calflgs}

            # Write back per-baseline scan-based flags:
            dp.read_exclude = ['data', 'time']
            for chunk_i, data in enumerate(iter(dp)):
                politsiyakat.log.info("Updating flags for chunk %d of %d..." %
                                      (chunk_i + 1, nchunk))
                politsiyakat.log.info("\tReading MS")
                for field_i, field_id in enumerate(source_scan_info.keys()):
                    politsiyakat.log.info("\tSelecting field %s..." %
                                          source_names[field_i])
                    has_updated = False
                    for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                        if chunk_i not in source_scan_info[field_id][s]["chunk_list"]:
                            continue
                        has_updated = True
                        flgs = new_flags[field_id][s].copy()
                        politsiyakat.log.info("\t\tUpdating flags for scan %d" % s)
                        # per antenna, all spw one go
                        epic_name = 'baseline flag update for %s' % source_names[field_i]
                        for bl in xrange(no_baselines):
                            politsiyakat.pool.submit_to_epic(epic_name,
                                                             _wk_per_bl_update,
                                                             bl,
                                                             field_id,
                                                             s,
                                                             si,
                                                             data,
                                                             flgs)
                        politsiyakat.pool.collect_epic(epic_name)
                    if not has_updated:
                        politsiyakat.log.info("\t\tNothing to be done for this field")
                if not simulate:
                    politsiyakat.log.info("\t\tWriting flag buffer back to disk...")
                    dp.flush_flags()

            # Dump a diagnostic plot of the number of bad phase channels per
            # baseline
            politsiyakat.log.info("Creating flag x uvdist plots:")
            uv_dist = uv_dist_per_baseline(no_baselines,
                                           nant,
                                           antenna_positions)
            ranked_uv_dist = np.argsort(uv_dist)
            for field_i, field_id in enumerate(fields):
                if field_id not in cal_fields:
                    continue
                politsiyakat.log.info("\tField %s:" % source_names[field_i])
                noplots = len(source_scan_info[field_id]["scan_list"])
                nxplts = int(np.ceil(np.sqrt(noplots)))
                nyplts = int(np.ceil(noplots / float(nxplts)))
                f, axarr = plt.subplots(nxplts,
                                        nyplts,
                                        dpi=dpi, figsize=(nxplts * plt_size, nyplts * plt_size))
                if not isinstance(axarr, np.ndarray):
                    arplt = np.empty((1,1), dtype=object)
                    arplt[0,0] = axarr
                    axarr = arplt
                axarr = axarr.reshape([nyplts, nxplts]) # ensure 2d array.. bug in mpl
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    politsiyakat.log.info("\t\tScan %s..." % s)

                    for c in xrange(ncorr):
                        axarr[si // nxplts, si % nxplts].plot(uv_dist[ranked_uv_dist],
                                                              np.sum(new_flags[field_id][s][ranked_uv_dist, :, c],
                                                                     axis=1),
                                                              label=ms_meta["ms_feed_names"][c])
                    axarr[si // nxplts, si % nxplts].set_title("scan %d" % s)
                    leg = axarr[si // nxplts, si % nxplts].legend()
                    leg.get_frame().set_alpha(0.5)
                f.text(0.0, 0.94, "FIELD: %s" % source_names[field_i], ha='left')
                f.text(0.5, 0.04, 'UVdist (m)', ha='center')
                f.text(0.04, 0.5, 'Number of channels flagged', va='center', rotation='vertical')
                f.savefig(output_dir + "/%s-FLAGGED_PHASE_UVDIST.CALFIELD_%s.png" %
                            (os.path.basename(ms),
                             source_names[field_i]))
                plt.close(f)

            # Create phase waterfall plots
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

            heatmaps = np.zeros([len(fields), no_baselines, ncorr, 128, nchan * nspw])
            fph = {}
            for field_i, field_id in enumerate(fields):
                sh = [len(source_scan_info[field_id]["scan_list"]), no_baselines, nchan * nspw, ncorr]

                if np.prod(np.array(sh)) == 0:
                    continue  # nothing to do
                fph[field_i] = np.zeros(sh)
                scan_mid = np.zeros([len(source_scan_info[field_id]["scan_list"])])
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    fph[field_i][si, :, :, :] = \
                        np.divide(source_scan_info[field_id][s]["sum_phase"],
                                  source_scan_info[field_id][s]["count_phase"])
                    scan_mid[si] = np.array([0.5 * (source_scan_info[field_id][s]["scan_end"] -
                                                    source_scan_info[field_id][s]["scan_start"]) - obs_start])

                for bl in xrange(no_baselines):
                    politsiyakat.pool.submit_to_epic("waterfall plots",
                                                     _wkr_bl_corr_regrid,
                                                     source_names[field_i],
                                                     field_i,
                                                     ncorr,
                                                     nchan,
                                                     nspw,
                                                     obs_end,
                                                     obs_start,
                                                     scan_mid,
                                                     fph,
                                                     antnames,
                                                     heatmaps,
                                                     bl)
                politsiyakat.pool.collect_epic("waterfall plots")
                politsiyakat.log.info("\t\t Done...")

                for corr in xrange(ncorr):
                    nxplts = int(np.ceil(np.sqrt(no_baselines)))
                    nyplts = int(np.ceil(no_baselines / float(nxplts)))
                    f, axarr = plt.subplots(nyplts,
                                            nxplts,
                                            dpi=dpi,
                                            figsize=(nyplts * plt_size, nxplts * plt_size),
                                            sharex=True, sharey=True)

                    scale_min = np.nanmin(heatmaps[field_i, :, corr, :, :])
                    scale_max = np.nanmax(heatmaps[field_i, :, corr, :, :])

                    for bl in xrange(no_baselines):
                        z = heatmaps[field_i, bl, corr, :, :]

                        im = axarr[bl // nxplts, bl % nxplts].imshow(
                            z,
                            aspect='auto',
                            extent=[0, nchan * nspw,
                                    0, (obs_end - obs_start) / 3600.0],
                            vmin=scale_min,
                            vmax=scale_max)
                        plt.colorbar(im, ax=axarr[bl // nxplts, bl % nxplts])
                        a1, a2 = ants_from_baseline(bl,nant)
                        a1n = antnames[a1]
                        a2n = antnames[a2]
                        axarr[bl // nxplts, bl % nxplts].set_title("%s & %s" % (a1n, a2n))
                    f.text(0.5, 0.94,
                           "FIELD: %s, CORR: %s" % (source_names[field_i], ms_meta["ms_feed_names"][corr]),
                           ha='center')
                    f.text(0.5, 0.04, 'channel', ha='center')
                    f.text(0.04, 0.5, 'Time (hrs)', va='center', rotation='vertical')
                    f.savefig(output_dir + "/%s-PHASE-FIELD-%s-CORR-%d.png" %
                              (os.path.basename(ms),
                               source_names[field_i],
                               corr))
                    plt.close(f)
            politsiyakat.log.info("\t\t Saved to %s" % output_dir)


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
            severe amplitude error.
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
            dpi = float(kwargs["plot_dpi"])
        except:
            dpi = 300
            politsiyakat.log.warn("Warning: defaulting to plot dpi of 300. Keyword 'plot_dpi' "
                                  "can be used to control this behaviour.")

        try:
            plt_size = float(kwargs["plot_size"])
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
                        politsiyakat.log.info("\t\t\tField %s is not present in this chunk" % source_names[field_i])
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
                            epic_name = 'antenna stats for %s' % source_names[field_i]
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
                                              (source_names[field_i],
                                               tot_flagged / tot_sel * 100.0))

            # Print some stats per field
            politsiyakat.log.info("Summary of flagging statistics per field:")
            for field_i, field_id in enumerate(fields):
                politsiyakat.log.info("\tField %s has the following scans:" % source_names[field_i])
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
                politsiyakat.log.info("\tField %s:" % source_names[field_i])
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
                politsiyakat.log.info("\tField %s:" % source_names[field_i])
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
                                          source_names[field_i])
                    has_updated = False
                    for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                        if chunk_i not in source_scan_info[field_id][s]["chunk_list"]:
                            continue
                        has_updated = True
                        politsiyakat.log.info("\t\tUpdating flags for scan %d" % s)
                        # per antenna, all spw one go
                        epic_name = 'antenna flag update for %s' % source_names[field_i]
                        for a in xrange(nant):
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

            heatmaps = np.zeros([len(fields), nant, ncorr, 512, nchan * nspw])
            famp = {}
            for field_i, field_id in enumerate(fields):
                sh = [len(source_scan_info[field_id]["scan_list"]), nant, nchan * nspw, ncorr]

                if np.prod(np.array(sh)) == 0:
                    continue # nothing to do
                famp[field_i] = np.zeros(sh)
                scan_mid =  np.zeros([len(source_scan_info[field_id]["scan_list"])])
                for si, s in enumerate(source_scan_info[field_id]["scan_list"]):
                    famp[field_i][si, :, :, :] = \
                        np.divide(source_scan_info[field_id][s]["tot_autopower"],
                                  source_scan_info[field_id][s]["tot_rowcount"])
                    scan_mid[si] = 0.5 * (source_scan_info[field_id][s]["scan_end"] -
                                          source_scan_info[field_id][s]["scan_start"]) - obs_start


                for ant in xrange(nant):
                    politsiyakat.pool.submit_to_epic("waterfall plots",
                                                     _wkr_ant_corr_regrid,
                                                     source_names[field_i],
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
                    nyplts = int(np.ceil(nant / float(nxplts)))
                    f, axarr = plt.subplots(nyplts,
                                            nxplts,
                                            dpi=dpi,
                                            figsize=(nyplts*plt_size, nxplts*plt_size),
                                            sharex=True, sharey=True)
                    dbheatmaps = 10 * np.log10(heatmaps[:, :, :, :, :])
                    #sanatize
                    dbheatmaps[dbheatmaps == -np.inf] = np.nan
                    dbheatmaps[dbheatmaps == np.inf] = np.nan
                    scale_min = np.nanmin(dbheatmaps[field_i, :, corr, :, :])
                    scale_max = np.nanmax(dbheatmaps[field_i, :, corr, :, :])
                    for ant in xrange(nant):
                        dbscale = dbheatmaps[field_i, ant, corr, :, :]

                        im = axarr[ant // nxplts, ant % nxplts].imshow(
                            dbscale,
                            aspect = 'auto',
                            extent = [0, nchan * nspw,
                                      0, (obs_end - obs_start) / 3600.0],
                            vmin = scale_min,
                            vmax = scale_max)
                        plt.colorbar(im, ax = axarr[ant // nxplts, ant % nxplts])
                        axarr[ant // nxplts, ant % nxplts].set_title(antnames[ant])
                    f.text(0.5, 0.94,
                           "FIELD: %s, CORR: %s (dB-scale)" % (source_names[field_i], ms_meta["ms_feed_names"][corr]),
                           ha='center')
                    f.text(0.5, 0.04, 'channel', ha='center')
                    f.text(0.04, 0.5, 'Time (hrs)', va='center', rotation='vertical')
                    f.savefig(output_dir + "/%s-AUTOCORR-FIELD-%s-CORR-%d.png" %
                                (os.path.basename(ms),
                                 source_names[field_i],
                                 corr))
                    plt.close(f)
            politsiyakat.log.info("\t\t Saved to %s" % output_dir)


######################################################
# WORKERS
######################################################
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
        scan_sel = data["scan"][chunk_start:chunk_end] == s
        ant_sel = np.logical_or(data["a1"][chunk_start:chunk_end] == a,
                                data["a2"][chunk_start:chunk_end] == a)
        accum_sel = np.logical_and(scan_sel, ant_sel)
        newf = np.logical_or(field_scan_flags_intra[field_id][si][a],
                             field_scan_flags_inter[field_id][si][a])
        data["flag"][np.argwhere(accum_sel)][chunk_start:chunk_end] = \
            np.logical_or(data["flag"][np.argwhere(accum_sel)][chunk_start:chunk_end],
                          newf)

def _wk_per_bl_update(bl,
                      field_id,
                      s,
                      si,
                      data,
                      flgs):

    nrow_per_chunk = 100
    nchunks = int(np.ceil(data["data"].shape[0] / float(nrow_per_chunk)))
    for n in xrange(nchunks):
        nrows_to_read = min(data["data"].shape[0] - (n * nrow_per_chunk),
                            nrow_per_chunk)
        chunk_start = n * nrow_per_chunk
        chunk_end = n * nrow_per_chunk + nrows_to_read
        scan_sel = data["scan"][chunk_start:chunk_end] == s
        bl_sel = data["baseline"][chunk_start:chunk_end] == bl
        accum_sel = np.logical_and(scan_sel, bl_sel)
        data["flag"][np.argwhere(accum_sel)][chunk_start:chunk_end] = \
            np.logical_or(data["flag"][np.argwhere(accum_sel)][chunk_start:chunk_end],
                          flgs[bl])


def _wk_per_ant_flag_stats(a,
                           ncorr,
                           nchan,
                           spw,
                           field_id,
                           s,
                           data):
    # get autocorrelation power per antenna
    totflagged = None
    totrowcount = None
    nrow_per_chunk = 100
    nchunks = int(np.ceil(data["data"].shape[0] / float(nrow_per_chunk)))
    for n in xrange(nchunks):
        nrows_to_read = min(data["data"].shape[0] - (n * nrow_per_chunk),
                            nrow_per_chunk)
        chunk_start = n * nrow_per_chunk
        chunk_end = n * nrow_per_chunk + nrows_to_read
        scan_sel = data["scan"][chunk_start:chunk_end] == s
        spw_sel = data["spw"][chunk_start:chunk_end] == spw
        accum_sel = np.logical_and(scan_sel, spw_sel)

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
    return ("antstat", a, spw, totflagged, totrowcount)

def _wk_per_bl_phase_stats(bl,
                           ncorr,
                           nchan,
                           spw,
                           field_id,
                           s,
                           data):
    # get autocorrelation power per antenna
    nbins = 120
    blchisq = np.zeros([nchan, ncorr])
    blsum = np.zeros([nchan, ncorr])
    blcount = np.zeros([nchan, ncorr])
    nrow_per_chunk = 100
    nchunks = int(np.ceil(data["data"].shape[0] / float(nrow_per_chunk)))
    for n in xrange(nchunks):
        nrows_to_read = min(data["data"].shape[0] - (n * nrow_per_chunk),
                            nrow_per_chunk)
        chunk_start = n * nrow_per_chunk
        chunk_end = n * nrow_per_chunk + nrows_to_read
        scan_sel = data["scan"][chunk_start:chunk_end] == s
        spw_sel = data["spw"][chunk_start:chunk_end] == spw
        accum_sel = np.logical_and(scan_sel, spw_sel)
        bl_sel = data["baseline"][chunk_start:chunk_end] == bl
        accum_sel = np.logical_and(accum_sel, bl_sel)
        accum_sel_tiled = np.tile(accum_sel, (ncorr, nchan, 1)).T
        # get flagging statistics per antenna
        sel = np.logical_and(accum_sel_tiled,
                             np.logical_not(data["flag"][chunk_start:chunk_end]))

        dn = data["data"][chunk_start:chunk_end].copy()
        dn[np.logical_not(sel)] = np.nan
        d = np.rad2deg(np.angle(dn))
        # expecting 0 rad phase angle for calibrators - can weigh by number of samples later when entire scan has
        # been processed
        blchisq += np.nansum(d**2, axis=0)
        blsum += np.nansum(d, axis=0)
        blcount += np.sum(sel, axis=0)
    return ("blstat", bl, spw, blchisq, blsum, blcount)

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
        scan_sel = data["scan"][chunk_start:chunk_end] == s
        spw_sel = data["spw"][chunk_start:chunk_end] == spw
        accum_sel = np.logical_and(scan_sel, spw_sel)
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
                          [1, famp[field_i].shape[0]]).flatten()
        heatmap = interp.griddata((xcoords,
                                   ycoords),
                                  famp[field_i][:, ant, :, corr].flatten(),
                                  (X[:, None], Y[None, :]),
                                  method='nearest')
        heatmaps[field_i, ant, corr, :, :] = heatmap

def _wkr_bl_corr_regrid(field_name,
                        field_i,
                        ncorr,
                        nchan,
                        nspw,
                        obs_end,
                        obs_start,
                        scan_mid,
                        fph,
                        antnames,
                        heatmaps,
                        bl):

    for corr in xrange(ncorr):
        X, Y = np.linspace(0.0, float(obs_end - obs_start), 128), \
               np.arange(nchan * nspw)
        xcoords = np.repeat(scan_mid, nchan * nspw)
        ycoords = np.tile(np.arange(nchan * nspw),
                          [1, fph[field_i].shape[0]]).flatten()
        heatmap = interp.griddata((xcoords,
                                   ycoords),
                                  fph[field_i][:, bl, :, corr].flatten(),
                                  (X[:, None], Y[None, :]),
                                  method='nearest')
        heatmaps[field_i, bl, corr, :, :] = heatmap
