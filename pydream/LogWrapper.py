import numpy as np

from pm4py.objects.log.log import EventLog
from pydream.util.Functions import time_delta_seconds


class LogWrapper:
    def __init__(self, log, resources=None, activity_key="concept:name", ts_key="time:timestamp"):
        """
        Creates a new instance of a LogWrapper. This class provides extra functionalities to instances of <class \'pm4py.objects.log.log.EventLog\'> for Decay Replay
        :param log: event log of class pm4py.objects.log.log.EventLog to be wrapped
        :param resources: resources to consider, pass as a list of keys
        """
        if not isinstance(log, EventLog):
            raise ValueError('The input event log is not an instance of <class \'pm4py.objects.log.log.EventLog\'>')

        self.activity_key = activity_key
        self.ts_key = ts_key

        self.log = log
        self.ignored_traces = set()
        self.max_trace_duration = self.getTraceDurationStats()["max"]

        self.iter_index = -1
        self.iter_remaining = len(self.log) - len(self.ignored_traces)


        """ SETUP RESOURCE COUNTER """
        if resources is None:
            self.resource_keys = None
        else:
            resource_values = set()
            self.resource_keys = dict()
            while self.iterator_hasNext():
                trace = self.iterator_next()
                for event in trace:
                    for resource_key in resources:
                        if resource_key in event.keys():
                            resource_values.add(resource_key + "_:_" + str(event[resource_key]))
            self.iterator_reset()
            resource_values = list(resource_values)
            for i in range(len(resource_values)):
                self.resource_keys[resource_values[i]] = i

    def getResourceKeys(self):
        return self.resource_keys

    def getTraceDurationStats(self):
        """ Statistics on trace durations in seconds """
        durations = list()

        for trace in self.log:
            if len(trace) < 2:
                self.ignored_traces.add(trace.attributes[self.activity_key])
            else:
                durations.append(time_delta_seconds(trace[0][self.ts_key], trace[-1][self.ts_key]))

        durations = list(filter(lambda a: a != 0.0, durations))

        mean = np.mean(durations)
        median = np.median(durations)
        upper_quartile = np.percentile(durations, 75)
        lower_quartile = np.percentile(durations, 25)
        min_d = np.min(durations)
        max_d = np.max(durations)

        self.stats = dict()
        self.stats["mean"] = mean
        self.stats["median"] = median
        self.stats["upper_quartile"] = upper_quartile
        self.stats["lower_quartile"] = lower_quartile
        self.stats["min"] = min_d
        self.stats["max"] = max_d

        #print(self.stats)
        return self.stats

    def isTraceIgnored(self, trace):
        return trace.attributes[self.activity_key] in self.ignored_traces

    def iterator_reset(self):
        self.iter_index = -1
        self.iter_remaining = len(self.log) - len(self.ignored_traces)

    def iterator_hasNext(self):
        return self.iter_remaining > 0

    def iterator_next(self):
        if self.iterator_hasNext():
            self.iter_index += 1
            trace = self.log[self.iter_index]

            while(self.isTraceIgnored(trace)):
                self.iter_index += 1
                trace = self.log[self.iter_index]

            self.iter_remaining -= 1
            return trace
        else:
            raise ValueError('No more traces in log iterator.')