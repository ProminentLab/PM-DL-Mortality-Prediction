import json
from converter.variants.LogBuilderMimic3_Tools import LogBuilderMimic3_Tools
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer

if __name__ == "__main__":
    with open('data/output/meta.json') as json_file:
        meta = json.load(json_file)

    for ending in ["train", "test", "val"]:
        print("** Creating XES for " + ending + " ***")
        with open('data/output/traces_proc_'+ending+'.json') as json_file:
            traces = json.load(json_file)
            print(len(traces))

        log_builder = LogBuilderMimic3_Tools()
        log = log_builder.build(traces=traces, metadata=meta)

        with open("data/output/log_"+ending+".xes", "w") as file:
            XesXmlSerializer().serialize(log, file)
