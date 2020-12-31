from pydream.LogWrapper import LogWrapper
from pydream.EnhancedPN import EnhancedPN

from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.objects.petri.exporter import pnml as pnml_exporter
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.algo.discovery.inductive import factory as inductive_miner

if __name__== "__main__":
    ## DISCOVER A PROCESS MODEL USING SPLIT MINER. THEN ENHANCE IT WITH THIS SCRIPT!

    config = ["LinearDecay", "LinearDecay_mean", "ExpDecay_max", "LogDecay_mean", "TokenCount", "Marking"]

    train_log = xes_import_factory.apply('data/output/log_train.xes')

    net, initial_marking, final_marking = inductive_miner.apply(train_log)
    pnml_exporter.export_net(net, initial_marking, "data/models/im_log_train.pnml")

    #net, initial_marking, final_marking = pnml_importer.import_net("data/models/sm_log_train.pnml")

    print("PN Places:", len(net.places))
    print("PN Transitions:", len(net.transitions))

    log_wrapper_train = LogWrapper(train_log)

    enhanced_pn = EnhancedPN(net, initial_marking, tss_settings=config)
    enhanced_pn.enhance(log_wrapper_train)

    enhanced_pn.saveToFile("data/models/im_log_train_multienhanced.json")



