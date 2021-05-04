import json

from pydream.LogWrapper import LogWrapper
from pydream.EnhancedPN import EnhancedPN

from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.petri.importer import pnml as pnml_importer

if __name__== "__main__":

    config = ["LinearDecay", "LinearDecay_mean", "ExpDecay_max", "LogDecay_mean", "TokenCount", "Marking"]
    count_events = True

    train_log = xes_import_factory.apply('data/output/log_train.xes')
    print(len(train_log))

    test_log = xes_import_factory.apply('data/output/log_test.xes')
    print(len(test_log))

    val_log = xes_import_factory.apply('data/output/log_val.xes')
    print(len(val_log))

    log_wrapper_train = LogWrapper(train_log)
    log_wrapper_test = LogWrapper(test_log)
    log_wrapper_val = LogWrapper(val_log)







    net, initial_marking, final_marking = pnml_importer.import_net("data/models/sm_log_train.pnml")
    enhanced_pn = EnhancedPN(net, initial_marking, decay_function_file="data/models/sm_log_train_multienhanced.json")

    print("PN Places:", len(net.places))
    print("PN Transitions:", len(net.transitions))

    with open('data/output/meta.json') as json_file:
        meta = json.load(json_file)

    enhanced_pn.prepareEventCounter()
    timedstatesamples, tss_objects = enhanced_pn.decay_replay(log_wrapper=log_wrapper_train, meta=meta, oversample=None, train=False, count_events=count_events)
    with open("data/output/sm_log_train_tss_train.json", 'w') as fp:
        json.dump(timedstatesamples, fp)

    timedstatesamples, tss_objects = enhanced_pn.decay_replay(log_wrapper=log_wrapper_test, meta=meta, oversample=None, train=False, count_events=count_events)
    with open("data/output/sm_log_train_tss_test.json", 'w') as fp:
        json.dump(timedstatesamples, fp)

    timedstatesamples, tss_objects = enhanced_pn.decay_replay(log_wrapper=log_wrapper_val, meta=meta, oversample=None,
                                                                  train=False, count_events=count_events)
    with open("data/output/sm_log_train_tss_val.json", 'w') as fp:
        json.dump(timedstatesamples, fp)