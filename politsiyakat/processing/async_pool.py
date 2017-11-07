from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from progress.bar import FillingCirclesBar

class async_pool:
    def __init__(self, n_ioworkers, n_comworkers, report_progress = True):
        self.__ppool = ThreadPoolExecutor(max_workers=n_comworkers)
        self.__iopool = ThreadPoolExecutor(max_workers=n_ioworkers)
        self.__ncomworkers = n_comworkers
        self.__nioworkers = n_ioworkers
        self.__epics = {}
        self.__repprog = report_progress
    @property
    def num_com_workers(self):
        return self.__ncomworkers

    @property
    def num_io_workers(self):
        return self.__nioworkers

    def submit_io(self, *args, **kwargs):
        """ Returns future to io work (see concurrent) """
        return self.__iopool.submit(*args, **kwargs)

    def submit_compute(self, *args, **kwargs):
        """ Returns future to io work (see concurrent) """
        return self.__ppool.submit(*args, **kwargs)

    def submit_to_epic(self, epic_name, *args, **kwargs):
        """ Submit a named job as part of a series. See also concurrent """
        fut = self.__ppool.submit(*args, **kwargs)
        if epic_name not in self.__epics.keys():
            self.__epics[epic_name] = [fut]
        else:
            self.__epics[epic_name].append(fut)

    def collect_epic(self, epic_name):
        """
            Collects all jobs in epic
            epic_name: name of job series to submit this job to
        :raise
            KeyError if epic name not registered before
        :return
            list of worker results
        """
        if epic_name not in self.__epics.keys():
            raise KeyError("Cannot find named epic '%s'" % epic_name)

        if self.__repprog:
            bar = FillingCirclesBar("Processing epic '%s'" % epic_name, max=len(self.__epics[epic_name]))
            bar.start()
            results = []
            j = 0
            while j < len(self.__epics[epic_name]):
                try:
                    results.append(self.__epics[epic_name][j].result(timeout=1))
                    self.__epics[epic_name].remove(self.__epics[epic_name][j])
                    if len(self.__epics[epic_name]) > 0:
                        j %= len(self.__epics[epic_name])
                    else:
                        j = 0
                    bar.next()
                except TimeoutError:
                    j = (j + 1) % len(self.__epics[epic_name])
            bar.finish()
        else:
            results = []
            for f in self.__epics[epic_name]:
                results.append(f.result())
            self.__epics[epic_name] = []
        return results

    def shutdown(self):
        """ Shuts down pools """
        self.__ppool.shutdown()
        self.__iopool.shutdown()