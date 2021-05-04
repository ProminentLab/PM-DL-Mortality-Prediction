from pydream.predictive.nap.SPLIT_auc import SPLIT

if __name__== "__main__":
    split = SPLIT(net=None,
                    tss_train_file="data/output/sm_log_train_tss_train.json",
                    tss_test_file="data/output/sm_log_train_tss_test.json",
                    tss_val_file="data/output/sm_log_train_tss_val.json"
                  )

    split.train(checkpoint_path="data/models/nap_ckpt", name="new_model", save_results=True)