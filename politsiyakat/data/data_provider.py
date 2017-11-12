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
import politsiyakat
import os
from politsiyakat.data.misc import *

'''
Enumeration of stokes and correlations used in MS2.0 - as per Stokes.h in casacore, the rest are left unimplemented:
These are useful when working with visibility data (https://casa.nrao.edu/Memos/229.html#SECTION000613000000000000000)
'''
stokes_types = ['Undef', 'I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY', 'YX', 'YY']

class data_provider:
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

        with table(ms + "::POLARIZATION", readonly=True, ack=False) as t:
            feeds = [stokes_types[t] for t in t.getcell("CORR_TYPE", 0)]
            ms_meta["ms_feed_names"] = feeds

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

    def __init__(self, msname, data_column, nrows_chunk, **kwargs):
        """
            MS data provider
        :param kwargs:
            "msname" : name of measurement set
            "data_column" : data column to use for amplitude flagging
            "nrows_chunk" : size of chunk in rows to read
        """
        self.__ms = msname
        self.__data_column = data_column
        self.__nrows_chunk = nrows_chunk
        self.__ichunk = 0
        kwargs["msname"] = msname
        kwargs["data_column"] = data_column
        kwargs["nrows_chunk"] = nrows_chunk
        self.__ms_meta = data_provider.check_ms(**kwargs)
        self.__nchunk = self.__ms_meta["nchunk"]
        self.__read_exclude = []  # everything by default
        self.__old_read_exclude = []  # everything by default
        self.__maintable_chunk = self.__alloc_buffer()
        self.__double_buffer = self.__alloc_buffer()
        self.__nxchunk_wkr = None
        self.__dbchunk_wkr = None
        self.__rows_read = 0

    @property
    def read_exclude(self):
        return self.__read_exclude

    @read_exclude.setter
    def read_exclude(self, value):
        if not isinstance(value, list):
            raise ValueError("readexclude list not list")
        self.__read_exclude = value

    @property
    def data(self):
        """
            Gets shallow copy of data buffer
        :return
            buffer if first chunk was read, None otherwise
        """
        if self.__maintable_chunk is not None:
            return {'a1': self.__maintable_chunk['a1'][:self.__rows_read],
                    'a2': self.__maintable_chunk['a1'][:self.__rows_read],
                    'baseline': self.__maintable_chunk['baseline'][:self.__rows_read],
                    'field': self.__maintable_chunk['field'][:self.__rows_read],
                    'flag': self.__maintable_chunk['flag'][:self.__rows_read],
                    'desc': self.__maintable_chunk['desc'][:self.__rows_read],
                    'data': self.__maintable_chunk['data'][:self.__rows_read],
                    'spw': self.__maintable_chunk['spw'][:self.__rows_read],
                    'scan': self.__maintable_chunk['scan'][:self.__rows_read],
                    'time': self.__maintable_chunk['time'][:self.__rows_read]}
        else:
            return None

    def flush_flags(self):
        """
            Stores flags of current chunk to MS
        """
        with table(self.__ms, readonly=False, ack=False) as t:
            rows_to_write = min(t.nrows() - ((self.__ichunk - 1) * self.__nrows_chunk),
                                self.__nrows_chunk)
            t.putcol("FLAG",
                     self.__maintable_chunk["flag"][:rows_to_write],
                     (self.__ichunk - 1) * self.__nrows_chunk,
                     rows_to_write)

    def __len__(self):
        return self.__nchunk

    def __iter__(self):
        if (self.__ichunk != self.__nchunk - 1) or \
           (len([v for v in self.__old_read_exclude if v not in self.__read_exclude]) != 0):
            self.__nxchunk_wkr = None
            self.__dbchunk_wkr = None
        self.__ichunk = 0
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass

    def __del__(self):
        pass

    def next(self):
        if self.__ichunk < self.__nchunk:
            if self.__nxchunk_wkr is None or self.__dbchunk_wkr is None:
                # first chunk will start the double buffering setup
                self.__nxchunk_wkr = politsiyakat.pool.submit_io(politsiyakat.data.data_provider._read_buffer,
                                                                self.__ms,
                                                                self.__read_exclude,
                                                                self.__maintable_chunk,
                                                                self.__nrows_chunk,
                                                                self.__ichunk,
                                                                self.__ms_meta,
                                                                self.__data_column
                                                               )
                self.__rows_read = self.__nxchunk_wkr.result()
                self.__old_read_exclude = self.__read_exclude
                self.__dbchunk_wkr = politsiyakat.pool.submit_io(politsiyakat.data.data_provider._read_buffer,
                                                                self.__ms,
                                                                self.__read_exclude,
                                                                self.__double_buffer,
                                                                self.__nrows_chunk,
                                                                (self.__ichunk + 1) % self.__nchunk,
                                                                self.__ms_meta,
                                                                self.__data_column)
            else:
                # swap buffers and start loading next buffer
                self.__rows_read = self.__dbchunk_wkr.result()
                self.__old_read_exclude = self.__read_exclude
                o_dbf = self.__maintable_chunk
                self.__maintable_chunk = self.__double_buffer
                self.__double_buffer = o_dbf
                self.__dbchunk_wkr = politsiyakat.pool.submit_io(politsiyakat.data.data_provider._read_buffer,
                                                                self.__ms,
                                                                self.__read_exclude,
                                                                self.__double_buffer,
                                                                self.__nrows_chunk,
                                                                (self.__ichunk + 1) % self.__nchunk,
                                                                self.__ms_meta,
                                                                self.__data_column)
            self.__ichunk += 1
            return self.data
        else:
            raise StopIteration()




    def __alloc_buffer(self):
        """
            Allocates a buffer to hold measurement set data
            Preallocating a buffer is far more efficient than
            the associated allocations and memcopies of getcol
        """
        vtype_map = {
            "bool" : np.bool,
            "boolean" : np.bool,
            "uchar" : np.uint8,
            "byte" : np.uint8,
            "short" : np.int16,
            "int" : np.int32,
            "uint" : np.uint32,
            "float" : np.float32,
            "double" : np.float64,
            "complex" : np.complex64,
            "dcomplex" : np.complex128,
            "string" : np.str
        }
        with table(self.__ms, readonly=True, ack=False) as t:
            desc = t.getcoldesc("ANTENNA1")
            a1 = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            desc = t.getcoldesc("ANTENNA2")
            a2 = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            baseline = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            desc = t.getcoldesc("FIELD_ID")
            field = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            desc = t.getcoldesc("FLAG")
            flag = np.empty((self.__nrows_chunk,
                             self.__ms_meta["nchan"],
                             self.__ms_meta["ncorr"]), dtype=vtype_map[desc["valueType"]])
            desc = t.getcoldesc("DATA_DESC_ID")
            ddesc = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            spw = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            desc = t.getcoldesc(self.__data_column)
            data = np.empty((self.__nrows_chunk,
                             self.__ms_meta["nchan"],
                             self.__ms_meta["ncorr"]), dtype=vtype_map[desc["valueType"]])
            desc = t.getcoldesc("SCAN_NUMBER")
            scan = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            desc = t.getcoldesc("TIME")
            time = np.empty((self.__nrows_chunk), dtype=vtype_map[desc["valueType"]])
            buf = {'a1': a1,
                   'a2': a2,
                   'baseline': baseline,
                   'field': field,
                   'flag': flag,
                   'desc': ddesc,
                   'data': data,
                   'spw': spw,
                   'scan': scan,
                   'time': time}
            return buf

def _read_buffer(ms,
                  read_exclude,
                  buffer,
                  nrows_chunk,
                  ichunk,
                  ms_meta,
                  data_column):
    """ Reads next buffer """
    map_descriptor_to_spw = ms_meta["map_descriptor_to_spw"]
    with table(ms, readonly=True, ack=False) as t:
        nrows_to_read = min(t.nrows() - (ichunk * nrows_chunk),
                            nrows_chunk)
        if not "a1" in read_exclude:
            t.getcolnp(
                "ANTENNA1",
                buffer["a1"][0:nrows_to_read],
                ichunk * nrows_chunk,
                nrows_to_read)

        if not "a2" in read_exclude:
            t.getcolnp(
                "ANTENNA2",
                buffer["a2"][0:nrows_to_read],
                ichunk * nrows_chunk,
                nrows_to_read)
        if not "a1" in read_exclude \
            and not "a2" in read_exclude:
            buffer["baseline"][:nrows_to_read] = \
                baseline_index(buffer["a1"],
                               buffer["a2"],
                               ms_meta["nant"])[:nrows_to_read]
        if not "field" in read_exclude:
            t.getcolnp(
                "FIELD_ID",
                buffer["field"][0:nrows_to_read],
                ichunk * nrows_chunk,
                nrows_to_read)
        if not "data" in read_exclude:
            t.getcolnp(
                data_column,
                buffer["data"][0:nrows_to_read,:,:],
                ichunk * nrows_chunk,
                nrows_to_read)
        if not "flag" in read_exclude:
            t.getcolnp(
                "FLAG",
                buffer["flag"][0:nrows_to_read,:,:],
                ichunk * nrows_chunk,
                nrows_to_read)
        if not "desc" in read_exclude:
            t.getcolnp(
                "DATA_DESC_ID",
                buffer["desc"][0:nrows_to_read],
                ichunk * nrows_chunk,
                nrows_to_read)
        if not "spw" in read_exclude \
            and not "desc" in read_exclude:
            buffer["spw"][:nrows_to_read] = \
                map_descriptor_to_spw[buffer["desc"]][:nrows_to_read]
        if not "scan" in read_exclude:
            t.getcolnp(
                "SCAN_NUMBER",
                buffer["scan"][0:nrows_to_read],
                ichunk * nrows_chunk,
                nrows_to_read)
        if not "time" in read_exclude:
            t.getcolnp(
                "TIME",
                buffer["time"][0:nrows_to_read],
                ichunk * nrows_chunk,
                nrows_to_read)
    return nrows_to_read
