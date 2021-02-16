import json

class TimedStateSample:
    def __init__(self, current_time, decay_values, token_counts, marking, place_list, tss_settings, resource_count=None, resource_indices=None, loadExisting=False):
        self.data = {'current_time' : current_time}
        self.data["tss_settings"] = tss_settings

        indices = len(tss_settings)
        hasMarking = False
        if "Marking" in tss_settings:
            indices -= 1
            hasMarking = True
        hasTokenCounter = False
        if "TokenCount" in tss_settings:
            indices -= 1
            hasTokenCounter = True

        if not loadExisting:
            decay_vector = []
            token_count_vector = []
            marking_vector = []

            for index in range(indices):
                values = list()
                for place in place_list:
                    values.append(decay_values[index][str(place)])
                decay_vector.append(values)

            for place in place_list:
                token_count_vector.append(token_counts[str(place)])

                if place in marking:
                    marking_vector.append(int(marking[place]))
                else:
                    marking_vector.append(0)

            self.data["TimedStateSample"] = [decay_vector]

            if hasTokenCounter:
                self.data["TimedStateSample"].append(token_count_vector)

            if hasMarking:
                self.data["TimedStateSample"].append(marking_vector)

            if resource_count is not None:
                resource_vector = [0 for i in range(len(resource_indices.keys()))]
                for key in resource_count.keys():
                    resource_vector[resource_indices[key]] = resource_count[key]
                self.data["TimedStateSample"].append(resource_vector)

        else:
            """ Load from File """
            self.data = {'current_time' : current_time,
                         'TimedStateSample' : [decay_values, token_counts, marking]}

    def setResourceVector(self, resource_vector):
        if len(self.data["TimedStateSample"]) < 4:
            self.data["TimedStateSample"].append(resource_vector)
        else:
            self.data["TimedStateSample"][3] = resource_vector

    def setRecentEvent(self, event):
        self.data["recentEvent"] = event

    def setNextEvent(self, event):
        self.data["nextEvent"] = event

    def setGender(self, gender):
        self.data["gender"] = gender

    def setAge(self, age):
        self.data["age"] = age

    def setPatient(self, patient):
        self.data["patient"] = patient

    def setCharlson(self, charlson):
        self.data["charlson"] = charlson

    def setComorbScores(self, comorb_scores):
        self.data["cs1"] = comorb_scores[0]
        self.data["cs2"] = comorb_scores[1]

    def setOasis(self, scores):
        self.data["oasis"] = scores

    def setSofa(self, scores):
        self.data["sofa"] = scores

    def setSapsii(self, scores):
        self.data["sapsii"] = scores

    def setApsiii(self, scores):
        self.data["apsiii"] = scores

    def setEventCount(self, eventcount):
        self.data["eventcount"] = eventcount

    def setElixhauser(self, elixhauser):
            self.data["elixhauser"] = elixhauser

    def setEthnicity(self, ethnicity):
        self.data["ethnicity"] = ethnicity
        if "white" in str(ethnicity).lower():
            self.data["ethnicity_enc"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        elif "black" in str(ethnicity).lower():
            self.data["ethnicity_enc"] = [0.0, 1.0, 0.0, 0.0, 0.0]
        elif "asian" in str(ethnicity).lower() or "middle eastern" in str(ethnicity).lower():
            self.data["ethnicity_enc"] = [0.0, 0.0, 1.0, 0.0, 0.0]
        elif "latino" in str(ethnicity).lower() or "hispanic" in str(ethnicity).lower():
            self.data["ethnicity_enc"] = [0.0, 0.0, 0.0, 1.0, 0.0]
        else:
            self.data["ethnicity_enc"] = [0.0, 0.0, 0.0, 0.0, 1.0]

    def setNextTimestamp(self, ts):
        self.data["nextTimestamp"] = ts

    def export(self):
        return self.data

def loadTimedStateSamples(filename):
    """
    Load decay functions for a given petri net from file.

    :param filename: filename of Timed State Samples
    :return: list containing TimedStateSample objects
    """
    final = list()
    with open(filename) as json_file:
        tss = json.load(json_file)
        for sample in tss:
            ts = TimedStateSample(sample["current_time"],
                             sample["TimedStateSample"][0],
                             sample["TimedStateSample"][1],
                             sample["TimedStateSample"][2],
                             None, loadExisting=True)
            """ Add resource count if exists """
            if len(sample["TimedStateSample"]) > 3:
                ts.setResourceVector(sample["TimedStateSample"][3])

            """ Add next event if present """
            if "nextEvent" in sample.keys():
                ts.setNextEvent(sample["nextEvent"])

            final.append(ts)
    return final