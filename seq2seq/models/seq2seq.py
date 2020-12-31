import json
import numpy as np
import keras.backend as K

from keras.layers import Input, Dense, Reshape,  Multiply, Dropout
from keras.models import Model
from keras.losses import mean_absolute_error
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils import class_weight

from keras.layers import LSTM as LSTM

class Seq2Seq:
    def __init__(self, dataLoader, options=None, train=True):
        self.dataLoader = dataLoader

        """ Options """
        self.opts = {"seed" : 664646,
                     "n_epochs" : 300,
                     "n_batch_size" : 128,
                     "dropout_rate" : 0.20,
                     "eval_size" : 0.0001,
                     "activation_function" : "relu"}

        if options is not None:
            for key in options.keys():
                self.opts[key] = options[key]

        """ Define Model Here """
        n_timesteps_in = self.dataLoader.X_in.shape[1]
        n_features = self.dataLoader.X_in.shape[2]
        n_classes = self.dataLoader.labels.shape[1]
        embedding = 30

        """ Encoder """
        enc_input = Input(shape=(n_timesteps_in, n_features), name="enc_input")
        enc_lay1 = LSTM(100, return_sequences=True, name="enc_lay1")(enc_input)
        enc_lay1= Dropout(self.opts["dropout_rate"])(enc_lay1)
        enc_lay2 = LSTM(embedding, return_sequences=False, name="enc_lstm1")(enc_lay1)
        enc_lay2 = Dropout(self.opts["dropout_rate"])(enc_lay2)
        enc_out = Dense(embedding,activation="sigmoid", name="enc_dense")(enc_lay2)
        self.encoder = Model(input=enc_input, output=enc_out, name="encoder")
        self.encoder.summary()

        """ Decoder """
        dec_mask_input = Input(shape=(n_timesteps_in, n_features), name="dec_mask_input")
        dec_fc1 = Dense(n_timesteps_in * embedding, activation="relu", name="dec_upscaling")(enc_out)
        dec_fc1 = Dropout(self.opts["dropout_rate"])(dec_fc1)
        dec_reshape = Reshape((n_timesteps_in, embedding), name="dec_reshape")(dec_fc1)
        dec_out = LSTM(100, return_sequences=True, name="dec_lay1")(dec_reshape)
        dec_out = Dropout(self.opts["dropout_rate"])(dec_out)
        dec_out = LSTM(n_features, return_sequences=True, name="dec_out_lstm")(dec_out)
        dec_out = Multiply(name="dec_out")([dec_mask_input,dec_out])
        self.encoder_decoder = Model(inputs=[enc_input, dec_mask_input], output=dec_out, name="encoder_decoder")
        self.encoder_decoder.summary()

        """ Classifier Model """
        dense_output = Dense(35, activation='relu', name='class_fc1')(enc_out)
        dense_output = Dropout(self.opts["dropout_rate"])(dense_output)
        dense_output = Dense(15, activation='relu', name='class_fc2')(dense_output)
        dense_output = Dropout(self.opts["dropout_rate"])(dense_output)
        dense_output = Dense(n_classes, activation='softmax', name='class_predictions')(dense_output)
        self.encoder_classifier = Model(input=enc_input, output=dense_output, name="encoder_classifier")
        self.encoder_classifier.summary()
        self.model = Model(inputs=[enc_input, dec_mask_input], outputs=[dec_out, dense_output])


        losses = {
            "class_predictions": "categorical_crossentropy",
            "dec_out": mean_absolute_error,
        }
        lossWeights = {
			"class_predictions": 1.0,
            "dec_out": 0.5
        }

        print("WEIGHTS", lossWeights)
        metrics = {
            "class_predictions": "accuracy",
            "dec_out": "mae"
        }
        self.model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=metrics)

        ### Callbacks ###
        ckpt_file = "data/models/checkpoints/seq2seq.hdf5"
        checkpoint = ModelCheckpoint(ckpt_file, monitor='val_auc_macro', verbose=1, save_best_only=True, mode='max')

        sample_weight = class_weight.compute_sample_weight('balanced', self.dataLoader.labels)

        factor = 1. / np.sqrt(2)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=20, mode='auto',
                                      factor=factor, cooldown=0, min_lr=1e-4, verbose=2)

        if train:
            hist = self.model.fit(x={
                              "dec_mask_input" : self.dataLoader.X_out_mask,
                               "enc_input" : self.dataLoader.X_in
                           },
                           y={
                               "class_predictions": self.dataLoader.labels,
                               "dec_out": self.dataLoader.X_out
                           },
                           batch_size=self.opts["n_batch_size"],
                           epochs=self.opts["n_epochs"],
                           sample_weight = {"class_predictions": sample_weight},
                           callbacks=[self.EvaluationCallback(self.dataLoader.val_X_in, self.dataLoader.val_X_out_mask, self.dataLoader.val_labels,
                                                              self.dataLoader.test_X_in, self.dataLoader.test_X_out_mask, self.dataLoader.test_labels), checkpoint, reduce_lr])

            with open(str("data/models/checkpoints/seq2seq_hist.json"), 'w') as outfile:
                json.dump(str(hist.history), outfile)

    def createEvents(self):
        ckpt_file = "data/models/checkpoints/seq2seq.hdf5"
        self.model.load_weights(ckpt_file)

        train_embedded = self.encoder.predict(self.dataLoader.X_in)
        train_admissions = {}
        for i, admission in enumerate(self.dataLoader.train_admissions, 0):
            train_admissions[admission] = []
            embedding = np.rint(train_embedded[i])
            for j in range(embedding.shape[0]):
                if embedding[j] == 1:
                    train_admissions[admission].append("event_" + str(j))
            if len(train_admissions[admission]) < 2:
                train_admissions[admission].append("event_none")

        test_embedded = self.encoder.predict(self.dataLoader.test_X_in)
        test_admissions = {}
        for i, admission in enumerate(self.dataLoader.test_admissions,0):
            test_admissions[admission] = []
            embedding = np.rint(test_embedded[i])
            for j in range(embedding.shape[0]):
                if embedding[j] == 1:
                    test_admissions[admission].append("event_" + str(j))
            if len(test_admissions[admission]) < 2:
                test_admissions[admission].append("event_none")

        val_embedded = self.encoder.predict(self.dataLoader.val_X_in)
        val_admissions = {}
        for i, admission in enumerate(self.dataLoader.val_admissions, 0):
            val_admissions[admission] = []
            embedding = np.rint(val_embedded[i])
            for j in range(embedding.shape[0]):
                if embedding[j] == 1:
                    val_admissions[admission].append("event_" + str(j))
            if len(val_admissions[admission]) < 2:
                val_admissions[admission].append("event_none")

        with open("data/output/seq2seq_embeddings_test.json", 'w') as outfile:
            json.dump(test_admissions, outfile)
        with open("data/output/seq2seq_embeddings_train.json", 'w') as outfile:
            json.dump(train_admissions, outfile)
        with open("data/output/seq2seq_embeddings_val.json", 'w') as outfile:
            json.dump(val_admissions, outfile)


    class EvaluationCallback(Callback):
        def __init__(self, val_X_in, val_X_out_mask, val_labels, test_X_in, test_X_out_mask, test_labels):
            self.X_test = test_X_in
            self.X_test_mask = test_X_out_mask
            self.Y_test = test_labels
            self.Y_test_int = np.argmax(self.Y_test, axis=1)

            self.X_val = val_X_in
            self.X_val_mask = val_X_out_mask
            self.Y_val = val_labels
            self.Y_val_int = np.argmax(self.Y_val, axis=1)

            self.test_accs = []
            self.val_accs = []
            self.losses = []

        def on_train_begin(self, logs={}):
            self.test_accs = []
            self.val_accs = []
            self.losses = []

        def on_epoch_end(self, epoch, logs={}):
            inputs = {
                "enc_input": self.X_test,
                "dec_mask_input": self.X_test_mask
            }
            y_pred = self.model.predict(inputs)
            y_pred = y_pred[1].argmax(axis=1)

            test_acc = accuracy_score(self.Y_test_int, y_pred, normalize=True)
            precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test_int, y_pred, average='macro',pos_label=None)
            auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="macro")

            print("Test accuracy:", test_acc)
            print("Test precision:", precision)
            print("Test recall:", recall)
            print("Test fscore:", fscore)
            print("Test AUC:", auc)

            logs['test_acc'] = test_acc
            logs['test_prec_macro'] = precision
            logs['test_rec_macro'] = recall
            logs['test_fscore_macro'] = fscore
            logs['test_auc_macro'] = auc


            inputs = {
                "enc_input": self.X_val,
                "dec_mask_input": self.X_val_mask
            }
            y_pred = self.model.predict(inputs)
            y_pred = y_pred[1].argmax(axis=1)

            val_acc = accuracy_score(self.Y_val_int, y_pred, normalize=True)
            precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_val_int, y_pred, average='macro',pos_label=None)
            auc = multiclass_roc_auc_score(self.Y_val_int, y_pred, average="macro")

            print("Val accuracy:", val_acc)
            print("Val precision:", precision)
            print("Val recall:", recall)
            print("Val fscore:", fscore)
            print("Val AUC:", auc)

            logs['val_acc'] = val_acc
            logs['val_prec_macro'] = precision
            logs['val_rec_macro'] = recall
            logs['val_fscore_macro'] = fscore
            logs['val_auc_macro'] = auc


def masked_mae(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)