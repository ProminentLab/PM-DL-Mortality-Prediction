import json
import numpy as np

from pm4py.algo.conformance.tokenreplay.versions.token_replay import *
from pm4py.objects.petri import semantics

from pydream.util.DecayFunctions import LinearDecay, ExponentialDecay, LogDecay, REGISTER
from pydream.util.Functions import time_delta_seconds, time_add_seconds
from pydream.util.TimedStateSamples import TimedStateSample

def loadComorbidities():
    lookup = dict()
    with open('data/util/admission_elix_sequences.txt') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line.replace("\n", "")
            array = line.split(",")
            hadm_id = array[0]
            if len(array) > 1:
                comorbidity = list()
                comorbidity.append(float(array[1]))
                comorbidity.append(float(array[2]))
                lookup[hadm_id] = comorbidity
    return lookup

def loadOasis():
    lookup = dict()
    with open('data/util/admission_oasis.txt') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line.replace("\n", "")
            array = line.split(",")
            hadm_id = array[0]
            if len(array) > 1:
                values = list()
                for val in array[1:]:
                    values.append(float(val))
                lookup[hadm_id] = values
    return lookup

def loadSofa():
    lookup = dict()
    with open('data/util/admission_sofa.txt') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line.replace("\n", "")
            array = line.split(",")
            hadm_id = array[0]
            if len(array) > 1:
                values = list()
                for val in array[1:]:
                    values.append(float(val))
                lookup[hadm_id] = values
    return lookup

def loadSapsii():
    lookup = dict()
    with open('data/util/admission_sapsii.txt') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line.replace("\n", "")
            array = line.split(",")
            hadm_id = array[0]
            if len(array) > 1:
                values = list()
                for val in array[1:]:
                    values.append(float(val))
                lookup[hadm_id] = values
    return lookup

def loadApsiii():
    lookup = dict()
    with open('data/util/admission_apsiii.txt') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line.replace("\n", "")
            array = line.split(",")
            hadm_id = array[0]
            if len(array) > 1:
                values = list()
                for val in array[1:]:
                    values.append(float(val))
                lookup[hadm_id] = values
    return lookup

class EnhancedPN:
    def __init__(self, net, initial_marking, decay_function_file=None, activity_key="concept:name", ts_key="time:timestamp", tss_settings=None):
        """
        Creates a new instance of an enhanced petri net
        :param net: petri net loaded from pm4py
        :param initial_marking: initial marking from pm4py
        :param decay_function_file: default=None, path to existing decay function file for the petri net
        """
        self.config = {}
        self.net = net
        self.initial_marking = initial_marking
        self.resource_keys = None

        if tss_settings is None:
            self.tss_settings = list()
            self.tss_settings.append("LinearDecay")
            self.tss_settings.append("TokenCount")
            self.tss_settings.append("Marking")
        else:
            self.tss_settings = tss_settings

        self.token_count = False
        if "TokenCount" in self.tss_settings:
            self.token_count = True

        self.marking = False
        if "Marking" in self.tss_settings:
            self.marking = True

        self.decay_functions = list()
        self.decay_function_indices = dict()
        index = 0
        for decay_function in self.tss_settings:
            if (decay_function != "TokenCount") and (decay_function != "Marking"):
                self.decay_function_indices[decay_function] = index
                self.decay_functions.append(dict())
                index += 1
        self.num_activation_functions = len(self.decay_function_indices.keys())

        if decay_function_file is not None:
            self.loadFromFile(decay_function_file)

        self.activity_key = activity_key
        self.ts_key = ts_key
        self.MAX_REC_DEPTH = 50

        self.place_list = list()
        for place in self.net.places:
            self.place_list.append(place)
        #self.place_list = sorted(self.place_list)

        self.trans_map = {}
        for t in self.net.transitions:
            if str(t.label)[-2:] == "_+":
                self.trans_map[str(t.label)[0:-2]] = t
            elif str(t.label)[-1] == "_" or str(t.label)[-1] == "+":
                self.trans_map[str(t.label)[0:-1]] = t
            else:
                self.trans_map[t.label] = t

    def prepareEventCounter(self):
        self.transition_one_hot = dict()
        cnt = 0
        for t in self.trans_map.keys():
            if t is not None:
                self.transition_one_hot[t] = cnt
                cnt += 1


    def enhance(self, log_wrapper):
        """
        Enhance a given petri net based on an event log.
        :param log_wrapper: Event log under consideration as LogWrapper
        :return:
        """

        for decay_function in self.tss_settings:
            if (decay_function != "TokenCount") and (decay_function != "Marking"):
                index = self.decay_function_indices[decay_function]

                if decay_function == "LinearDecay":

                    """ Standard Enhancement """
                    beta = float(1)
                    reactivation_deltas = {}
                    for place in self.net.places:
                        reactivation_deltas[str(place)] = list()

                    log_wrapper.iterator_reset()
                    last_activation = {}
                    while (log_wrapper.iterator_hasNext()):
                        trace = log_wrapper.iterator_next()

                        for place in self.net.places:
                            if place in self.initial_marking:
                                last_activation[str(place)] = trace[0][self.ts_key]
                            else:
                                last_activation[str(place)] = -1

                        """ Replay and estimate parameters """
                        places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(self.net, self.MAX_REC_DEPTH)
                        marking = copy(self.initial_marking)

                        for event in trace:

                            if str(event[self.activity_key]) in self.trans_map.keys():
                                activated_places = []
                                toi = self.trans_map[str(event[self.activity_key])]

                                """ If Transition of interest is not enabled yet, then go through hidden"""
                                if not semantics.is_enabled(toi, self.net, marking):

                                    _, _, act_trans, _ = apply_hidden_trans(toi, self.net, copy(marking),
                                                                            places_shortest_path_by_hidden, [], 0, set(),
                                                                            [copy(marking)])
                                    for act_tran in act_trans:
                                        for arc in act_tran.out_arcs:
                                            activated_places.append(arc.target)
                                        marking = semantics.execute(act_tran, self.net, marking)

                                """ If Transition of interest is STILL not enabled yet, then naively add missing token to fulfill firing rule"""
                                if not semantics.is_enabled(toi, self.net, marking):
                                    for arc in toi.in_arcs:
                                        if arc.source not in marking:
                                            marking[arc.source] += 1

                                """ Fire transition of interest """
                                for arc in toi.out_arcs:
                                    activated_places.append(arc.target)
                                marking = semantics.execute(toi, self.net, marking)


                                """ Marking is gone - transition could not be fired ..."""
                                if marking is None:
                                    raise ValueError("Invalid Marking - Transition " + toi + " could not be fired.")

                                """ Update Time Recordings """
                                for activated_place in activated_places:
                                    if last_activation[str(activated_place)] != -1:
                                        time_delta = time_delta_seconds(last_activation[str(activated_place)],
                                                                        event[self.ts_key])
                                        if time_delta > 0:
                                            reactivation_deltas[str(place)].append(time_delta)
                                    last_activation[str(activated_place)] = event[self.ts_key]

                    """ Calculate decay function parameter """
                    for place in self.net.places:
                        if len(reactivation_deltas[str(place)]) > 1:
                            self.decay_functions[index][str(place)] = LinearDecay(alpha=1 / np.mean(reactivation_deltas[str(place)]),
                                                                           beta=beta)
                        else:
                            self.decay_functions[index][str(place)] = LinearDecay(alpha=1 / log_wrapper.max_trace_duration, beta=beta)

                if decay_function == "LinearDecay_max":
                    for place in self.net.places:
                            self.decay_functions[index][str(place)] = LinearDecay(alpha=1/log_wrapper.max_trace_duration, beta=1.0)

                if decay_function == "LinearDecay_mean":
                    for place in self.net.places:
                            self.decay_functions[index][str(place)] = LinearDecay(alpha=1/log_wrapper.getTraceDurationStats()["mean"], beta=1.0)

                if decay_function == "ExpDecay_max":
                    for place in self.net.places:
                            self.decay_functions[index][str(place)] = ExponentialDecay(max=log_wrapper.max_trace_duration)

                if decay_function == "ExpDecay_mean":
                    for place in self.net.places:
                            self.decay_functions[index][str(place)] = ExponentialDecay(max=log_wrapper.getTraceDurationStats()["mean"])

                if decay_function == "LogDecay_max":
                    for place in self.net.places:
                            self.decay_functions[index][str(place)] = LogDecay(alpha=1/log_wrapper.max_trace_duration)

                if decay_function == "LogDecay_mean":
                    for place in self.net.places:
                            self.decay_functions[index][str(place)] = LogDecay(alpha=1/log_wrapper.getTraceDurationStats()["mean"])

        """ Get resource keys to store """
        self.resource_keys = log_wrapper.getResourceKeys()

    def decay_replay(self, log_wrapper, meta, resources=None, oversample=None, train=False, count_events=False):
        """
        Decay Replay on given event log.
        :param log_wrapper: Input event log as LogWrapper to be replayed.
        :param resources: Resource keys to count (must have been counted during Petri net enhancement already!), as a list
        :return: list of timed state samples as JSON, list of timed state sample objects
        """

        comorb_scores_lookup = loadComorbidities()
        oasis_lookup = loadOasis()
        sofa_lookup = loadSofa()
        sapsii_lookup = loadSapsii()
        apsiii_lookup = loadApsiii()

        tss = list()
        tss_objs = list()

        decay_values = []
        for _ in range(len(self.decay_function_indices.keys())):
            decay_values.append(dict())

        token_counts = {}
        marks = {}

        last_activation = {}

        """ Initialize Resource Counter """
        count_resources = False
        resource_counter = None
        if log_wrapper.resource_keys is not None:
            count_resources = True
            resource_counter = dict()
            for key in log_wrapper.resource_keys.keys():
                resource_counter[key] = 0
        """ ---> """

        log_wrapper.iterator_reset()

        not_found_cnt = 0

        count_traces = 0
        while log_wrapper.iterator_hasNext():
            print("Count: " + str(count_traces))
            count_traces += 1

            trace = log_wrapper.iterator_next()

            """ prepare event counter """
            event_counter = list()
            for _ in self.transition_one_hot.keys():
                event_counter.append(0)
            """ ----> """

            isDead = True
            if str(trace.attributes['label']) == "False":
                isDead = False

            patient = trace.attributes["concept:name"]
            ethnicity = trace.attributes["ethnicity"]


            predict_cnt = 0

            resource_count = copy(resource_counter)
            """ Reset all counts for the next trace """
            for place in self.net.places:
                if place in self.initial_marking:
                    last_activation[str(place)] = trace[0][self.ts_key]
                else:
                    last_activation[str(place)] = -1

            for index in range(len(self.decay_function_indices.keys())):
                for place in self.net.places:
                    decay_values[index][str(place)] = 0.0

            for place in self.net.places:
                token_counts[str(place)] = 0.0
                marks[str(place)] = 0.0
            """ ----------------------------------> """

            places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(self.net, self.MAX_REC_DEPTH)
            marking = copy(self.initial_marking)

            """ Initialize counts based on initial marking """
            for place in marking:
                for index in range(len(self.decay_function_indices.keys())):
                    decay_values[index][str(place)] = self.decay_functions[index][str(place)].decay(t=0)
                token_counts[str(place)] += 1
                marks[str(place)] = 1
            """ ----------------------------------> """

            """ Replay """
            #time_past = None
            time_recent = None
            init_time = None


            for event_id in range(len(trace)):
                event = trace[event_id]
                if event_id == 0:
                    init_time = event[self.ts_key]

                time_past = time_recent
                time_recent = event[self.ts_key]

                if str(event[self.activity_key]) in self.trans_map.keys():
                    """ count up event """
                    if count_events:
                        event_cnt_index = self.transition_one_hot[str(event[self.activity_key])]
                        event_counter[event_cnt_index] = event_counter[event_cnt_index] + 1
                    """ ----> """

                    activated_places = list()

                    toi = self.trans_map[str(event[self.activity_key])]
                    """ If Transition of interest is not enabled yet, then go through hidden"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        _, _, act_trans, _ = apply_hidden_trans(toi, self.net, copy(marking),
                                                                places_shortest_path_by_hidden, [], 0, set(),
                                                                [copy(marking)])
                        for act_tran in act_trans:
                            for arc in act_tran.out_arcs:
                                activated_places.append(arc.target)
                            marking = semantics.execute(act_tran, self.net, marking)
                    """ If Transition of interest is STILL not enabled yet, then naively add missing token to fulfill firing rule"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        for arc in toi.in_arcs:
                            if arc.source not in marking:
                                marking[arc.source] += 1
                    """ Fire transition of interest """
                    for arc in toi.out_arcs:
                        activated_places.append(arc.target)
                    marking = semantics.execute(toi, self.net, marking)
                    """ Marking is gone - transition could not be fired ..."""
                    if marking is None:
                        raise ValueError("Invalid Marking - Transition " + toi + " could not be fired.")
                    """ ----->"""

                    """ Update Time Recordings """
                    for activated_place in activated_places:
                        last_activation[str(activated_place)] = event[self.ts_key]

                    """ Count Resources"""
                    if count_resources and resources is not None:
                        for resource_key in resources:
                            if resource_key in event.keys():
                                val = resource_key + "_:_" + event[resource_key]
                                if val in resource_count.keys():
                                    resource_count[val] += 1

                    """ Update Vectors and create TimedStateSamples """
                    if not time_past is None:
                        decay_values, token_counts = self.updateVectors(decay_values=decay_values,
                                                                        last_activation=last_activation,
                                                                        token_counts=token_counts,
                                                                        activated_places=activated_places,
                                                                        current_time=time_recent)

                        next_event_id = self.findNextEventId(event_id, trace)
                        if next_event_id is not None:
                            next_event = trace[next_event_id][self.activity_key]
                            next_ts = trace[next_event_id][self.ts_key]
                        else:
                            next_event = None
                            next_ts = None


                        if count_resources:
                            timedstatesample = TimedStateSample(time_delta_seconds(init_time, time_recent),
                                                                copy(decay_values), copy(token_counts), copy(marking),
                                                                copy(self.place_list), tss_settings=self.tss_settings,
                                                                resource_count=copy(resource_count),
                                                                resource_indices=log_wrapper.getResourceKeys())
                        else:
                            timedstatesample = TimedStateSample(time_delta_seconds(init_time, time_recent),
                                                                copy(decay_values), copy(token_counts), copy(marking),
                                                                copy(self.place_list), tss_settings=self.tss_settings)
                        timedstatesample.setNextEvent(next_event)

                        if next_ts is None:
                            next_time = 0.0
                        else:
                            next_time = time_delta_seconds(time_recent, next_ts)
                        timedstatesample.setNextTimestamp(next_time)

                    if ("PREDICT" in event.keys()):
                        if event["PREDICT"] == "PREDICT" or event["PREDICT"] is True or event["PREDICT"] == "True":
                            if trace.attributes["gender"] == "M":
                                timedstatesample.setGender(1)
                            else:
                                timedstatesample.setGender(0)
                            timedstatesample.setAge(float(trace.attributes["age"]))
                            timedstatesample.setEthnicity(ethnicity)

                            timedstatesample.setPatient(patient)



                            admission_id = str(int(event['admission_id']))
                            """ OASIS SCORES """
                            try:
                                if admission_id in oasis_lookup.keys():
                                    print(oasis_lookup[admission_id])
                                    timedstatesample.setOasis(oasis_lookup[admission_id])
                                else:
                                    empty = [0] * 10
                                    timedstatesample.setOasis(empty)
                            except:
                                empty = [0] * 10
                                timedstatesample.setOasis(empty)

                            """ SOFA SCORES """
                            try:
                                if admission_id in sofa_lookup.keys():
                                    timedstatesample.setSofa(sofa_lookup[admission_id])
                                else:
                                    empty = [0] * 7
                                    timedstatesample.setSofa(empty)
                            except:
                                empty = [0] * 7
                                timedstatesample.setSofa(empty)

                            """ SAPSII SCORES """
                            try:
                                if admission_id in sapsii_lookup.keys():
                                    timedstatesample.setSapsii(sapsii_lookup[admission_id])
                                else:
                                    empty = [0] * 16
                                    timedstatesample.setSapsii(empty)
                            except:
                                empty = [0] * 16
                                timedstatesample.setSapsii(empty)

                            """ APSIII SCORES """
                            try:
                                if admission_id in apsiii_lookup.keys():
                                    timedstatesample.setApsiii(apsiii_lookup[admission_id])
                                else:
                                    empty = [0] * 13
                                    timedstatesample.setApsiii(empty)
                            except:
                                empty = [0] * 13
                                timedstatesample.setApsiii(empty)

                            if count_events:
                                timedstatesample.setEventCount(event_counter)

                            if isDead:
                                timedstatesample.setNextEvent("True")
                            else:
                                timedstatesample.setNextEvent("False")

                            predict_cnt += 1

                            tss.append(timedstatesample.export())
                            tss_objs.append(timedstatesample)

                            print(timedstatesample.export()["TimedStateSample"][-1])

                            print("added")
                    else:
                        not_found_cnt += 1
        print("Number of tss:", len(tss_objs))
        print("Not found count of tss:", not_found_cnt)
        return tss, tss_objs

    def updateVectors(self, decay_values, last_activation, token_counts, activated_places, current_time):
        """ Update Decay Values """
        for index in range(len(self.decay_function_indices.keys())):
            for place in self.net.places:
                if last_activation[str(place)] == -1:
                    decay_values[index][str(place)] = 0.0
                else:
                    delta = time_delta_seconds(last_activation[str(place)], current_time)
                    decay_values[index][str(place)] = self.decay_functions[index][str(place)].decay(delta)

        """ Update Token Counts """
        for place in activated_places:
            token_counts[str(place)] += 1

        return decay_values, token_counts

    def findNextEventId(self, current_id, trace):
        next_event_id = current_id + 1

        found = False
        while (next_event_id < len(trace) and not found):
            event = trace[next_event_id]
            if str(event[self.activity_key]) in self.trans_map.keys():
                found = True
            else:
                next_event_id += 1

        if found == False:
            return None
        else:
            return next_event_id

    def saveToFile(self, file):
        """
        Save the decay functions of the EnhancedPN to file.
        :param file: Output file
        :return:
        """
        output = dict()
        dumping = copy(self.decay_functions)

        for index in range(len(dumping)):
            dump = dumping[index]
            for key in dump.keys():
                dumping[index][key] = dump[key].toJSON()
        output["decayfunctions"] = dumping
        output["resource_keys"] = self.resource_keys
        output["decay_function_indices"] = self.decay_function_indices
        output["tss_settings"] = self.tss_settings
        output["Marking"] = self.marking
        output["TokenCount"] = self.token_count

        with open(file, 'w') as fp:
            json.dump(output, fp)

    def loadFromFile(self, file):
        """
        Load decay functions for a given petri net from file.
        :param file: Decay function file
        :return:
        """
        with open(file) as json_file:
            decay_data = json.load(json_file)

        self.marking = decay_data["Marking"]
        self.token_count = decay_data["TokenCount"]
        self.decay_function_indices = decay_data["decay_function_indices"]
        self.tss_settings = decay_data["tss_settings"]

        self.decay_functions = []
        for _ in range(len(self.decay_function_indices.keys())):
            self.decay_functions.append(dict())

        for index in range(len(self.decay_function_indices.keys())):
            for place in self.net.places:
                self.decay_functions[index][str(place)] = None

        try:
            if not set(decay_data["decayfunctions"][0].keys()) == set(self.decay_functions[0].keys()):
                self.decay_functions = {}
                raise ValueError(
                    "Set of decay functions is not equal to set of places of the petri net. Was the decay function file build on the same petri net? Loading from file cancelled.")
        except:
            self.decay_functions = {}
            pass

        for key_index in self.decay_function_indices.keys():
            index = self.decay_function_indices[key_index]
            for place in decay_data["decayfunctions"][index].keys():
                DecayFunctionClass = REGISTER[decay_data["decayfunctions"][index][place]['DecayFunction']]
                df = DecayFunctionClass()
                df.loadFromDict(decay_data["decayfunctions"][index][place])
                self.decay_functions[index][str(place)] = df

        self.resource_keys = decay_data["resource_keys"]