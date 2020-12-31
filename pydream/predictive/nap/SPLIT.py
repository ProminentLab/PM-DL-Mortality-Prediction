from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
from keras.layers import Dropout, Dense, BatchNormalization
from pydream.util.TimedStateSamples import TimedStateSample
import itertools
from itertools import chain
from sklearn.utils import class_weight
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Dense, Input, Lambda, Conv2D, Flatten, Reshape, Permute
from keras.models import model_from_json, Model
import tensorflow as tf
from keras.layers import concatenate

def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def auc_loss(y_true, y_pred):
    """ ROC AUC Score.

    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.

    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.

    Measures overall performance for a full range of threshold levels.

    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.

    """
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

    pos = tf.expand_dims(pos, 0)
    neg = tf.expand_dims(neg, 1)

    # original paper suggests performance is robust to exact parameter choice
    gamma = 0.2
    p     = 3

    difference = tf.zeros_like(pos * neg) + pos - neg - gamma

    masked = tf.boolean_mask(difference, difference < 0.0)

    return tf.reduce_sum(tf.pow(-masked, p))

class SPLIT:
    def __init__(self, net, tss_train_file=None, tss_test_file=None, tss_val_file=None, options=None):
        """ Options """

        self.opts = {"seed" : 664646,
                     "n_epochs" : 300,
                     "n_batch_size" : 512,
                     "dropout_rate" : 0.2,
                     "activation_function" : "relu"}
        self.setSeed()

        if options is not None:
            for key in options.keys():
                self.opts[key] = options[key]

        """ Load data and setup """
        if tss_train_file is not None and tss_test_file is not None:
            self.X_train, self.X2_train, self.X3_train, self.Y_train = self.loadData(tss_train_file, train=True, filter=None)
            self.X_val, self.X2_val, self.X3_val, self.Y_val = self.loadData(tss_val_file, train=True, filter=None)
            self.X_test, self.X2_test, self.X3_test, self.Y_test = self.loadData(tss_test_file, train=False, filter=True)

            print(self.X_train.shape)
            print(self.X_val.shape)
            print(self.X_test.shape)

            self.X_train = np.zeros(shape=self.X_train.shape)
            #self.X2_train = np.zeros(shape=self.X2_train.shape)
            self.X3_train = np.zeros(shape=self.X3_train.shape)

            self.X_val = np.zeros(shape=self.X_val.shape)
            #self.X2_val = np.zeros(shape=self.X2_val.shape)
            self.X3_val = np.zeros(shape=self.X3_val.shape)

            self.X_test = np.zeros(shape=self.X_test.shape)
            #self.X2_test = np.zeros(shape=self.X2_test.shape)
            self.X3_test = np.zeros(shape=self.X3_test.shape)

            self.oneHotEncoderSetup()
            self.Y_train = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_train).reshape(-1, 1)))

            self.Y_test = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_test).reshape(-1, 1)))

            self.Y_val = np.asarray(
                self.onehot_encoder.transform(self.label_encoder.transform(self.Y_val).reshape(-1, 1)))

            ratio = 1.00
            self.X_train = np.concatenate([self.X_train, self.X_val[:int(ratio*len(self.X_val)),]])
            self.X2_train = np.concatenate([self.X2_train, self.X2_val[:int(ratio*len(self.X_val)),]])
            self.X3_train = np.concatenate([self.X3_train, self.X3_val[:int(ratio*len(self.X_val)),]])
            self.Y_train = np.concatenate([self.Y_train, self.Y_val[:int(ratio*len(self.X_val)),]])

            insize = self.X_train.shape[1]
            insize_meta = self.X2_train.shape[1]
            insize_eventcount = self.X3_train.shape[1]
            outsize = len(self.Y_train[0])

            learning_rate=1e-3 #was 1e-3
            optm = Adam(lr=learning_rate)

            # define two sets of inputs
            inputA = Input(shape=(insize,))
            inputB = Input(shape=(insize_meta,))
            inputC = Input(shape=(insize_eventcount,))

            # the first branch operates on the first input
            x = Dense(500, activation=self.opts["activation_function"])(inputA)
            x = BatchNormalization()(x)
            x = Dropout(self.opts["dropout_rate"])(x)
            x = Dense(250, activation=self.opts["activation_function"])(x)
            x = Dropout(self.opts["dropout_rate"])(x)
            x = Dense(75, activation=self.opts["activation_function"])(x)
            x = Dropout(self.opts["dropout_rate"])(x)
            x = Model(inputs=inputA, outputs=x)

            # the second branch opreates on the second input
            y = Dense(32, activation="relu")(inputB) #32
            y = BatchNormalization()(y)
            y = Dropout(self.opts["dropout_rate"])(y)
            y = Dense(16, activation="relu")(y) #16
            y = Dropout(self.opts["dropout_rate"])(y)
            y = Dense(8, activation="relu")(y)
            y = Model(inputs=inputB, outputs=y)

            # the second branch opreates on the second input
            w = Dense(55, activation="relu")(inputC)
            w = BatchNormalization()(w)
            w = Dropout(self.opts["dropout_rate"])(w)
            w = Dense(32, activation="relu")(w)
            w = Dropout(self.opts["dropout_rate"])(w)
            w = Dense(16, activation="relu")(w)
            w = Model(inputs=inputC, outputs=w)


            # combine the output of the branches
            combined = concatenate([x.output, y.output, w.output])

            z = Dense(32, activation="relu")(combined)
            z = Dense(16, activation="relu")(z)
            z = Dense(outsize, activation='softmax')(z)

            self.model = Model(inputs=[x.input, y.input, w.input], outputs=z)
            self.model.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])
            self.model.summary()

    def train(self, checkpoint_path, name, save_results=False):
        event_dict_file = str(checkpoint_path) + "/" + str(name) + "_split_onehotdict.json"
        with open(str(event_dict_file), 'w') as outfile:
            json.dump(self.one_hot_dict, outfile)

        with open(checkpoint_path + "/" + name + "_split_model.json", 'w') as f:
            f.write(self.model.to_json())

        ckpt_file = str(checkpoint_path) + "/" + str(name) + "_split_weights-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(ckpt_file, monitor='val_loss', verbose=0, save_best_only=False, mode='auto', save_weights_only=False, period=1)

        sample_weight = class_weight.compute_sample_weight('balanced', self.Y_train)

        factor = 1. / np.sqrt(2)

        le = LabelEncoder()
        y_ind = le.fit_transform(self.Y_train.ravel())

        print("Class weights : ", class_weight)

        hist = self.model.fit([self.X_train, self.X2_train, self.X3_train], [self.Y_train], sample_weight=sample_weight,
                            batch_size=self.opts["n_batch_size"], epochs=self.opts["n_epochs"], shuffle=True,
                            validation_data=([self.X_val, self.X2_val, self.X3_val], [self.Y_val]),
                            callbacks=[self.EvaluationCallback(self.X_test, self.X2_test, self.X3_test, self.Y_test, self.X_val, self.X2_val, self.X3_val, self.Y_val),
                                       checkpoint])

        if save_results:
            results_file = str(checkpoint_path) + "/" + str(name) + "_split_results.json"
            with open(str(results_file), 'w') as outfile:
                json.dump(str(hist.history), outfile)

    def train_on_val(self, checkpoint_path, name, save_results=False):
        event_dict_file = str(checkpoint_path) + "/" + str(name) + "_split_onehotdict.json"
        with open(str(event_dict_file), 'w') as outfile:
            json.dump(self.one_hot_dict, outfile)

        with open(checkpoint_path + "/" + name + "_split_model.json", 'w') as f:
            f.write(self.model.to_json())

        ckpt_file = str(checkpoint_path) + "/" + str(name) + "_split_weights.hdf5"
        checkpoint = ModelCheckpoint(ckpt_file, monitor='val_auc', verbose=1, save_best_only=True, mode='max')

        sample_weight = class_weight.compute_sample_weight('balanced', self.Y_val)

        hist = self.model.fit([self.X_val, self.X2_val, self.X3_val], [self.Y_val], sample_weight=sample_weight,
                            batch_size=self.opts["n_batch_size"], epochs=10, shuffle=True,
                            validation_data=([self.X_val, self.X2_val, self.X3_val], [self.Y_val]),
                            callbacks=[self.EvaluationCallback(self.X_test, self.X2_test, self.X3_test, self.Y_test, self.X_val, self.X2_val, self.X3_val, self.Y_val), checkpoint]) #, reduce_lr])

        if save_results:
            results_file = str(checkpoint_path) + "/" + str(name) + "_split_results.json"
            with open(str(results_file), 'w') as outfile:
                json.dump(str(hist.history), outfile)


    def oneHotEncoderSetup(self, net=["True", "False"]):
        """ Events to One Hot"""
        events = []
        for t in net:
            events.append(t)
        events = np.array(events)

        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(events)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(integer_encoded)

        self.one_hot_dict = {}
        for event in events:
            self.one_hot_dict[event] = list(self.onehot_encoder.transform([self.label_encoder.transform([event])])[0])

    def loadData(self, file, train, filter=None):

        x, x2, x3, y  = [], [], [], []
        patients = []

        with open(file) as json_file:
            tss = json.load(json_file)
            for sample in tss:
                if sample["nextEvent"] is not None :
                    decays = list(chain.from_iterable(sample["TimedStateSample"][0]))
                    mark = []
                    tcount = []
                    if "Marking" in sample['tss_settings']:
                        mark = sample["TimedStateSample"][1]
                        if "TokenCount" in sample['tss_settings']:
                            tcount = sample["TimedStateSample"][2]
                    elif ("TokenCount" in sample['tss_settings']) and ("Marking" not in sample['tss_settings']):
                            tcount = sample["TimedStateSample"][1]

                    x2s = list()
                    x2s.append((float(sample["age"])/100))
                    x2s.append(float(sample["gender"]))

                    for val in sample["ethnicity_enc"]:
                        x2s.append(float(val))

                    cnt = 0
                    for val in sample["oasis"]:
                        if cnt < 10:
                            x2s.append(float(val))
                        cnt += 1

                    cnt = 0
                    for val in sample["sofa"]:
                        if cnt < 7:
                            x2s.append(float(val))
                        cnt += 1

                    for val in sample["sapsii"]:
                        x2s.append(float(val))


                    for val in sample["apsiii"]:
                        x2s.append(float(val))

                    x2.append(x2s)

                    x3.append(sample["eventcount"])

                    if filter is None or filter == True:
                        x.append(list(itertools.chain(decays, mark, tcount)))
                        y.append(sample["nextEvent"])
                        patients.append(sample["patient"])
                    else:
                        if sample["patient"] in filter:
                            x.append(list(itertools.chain(decays, mark, tcount)))
                            y.append(sample["nextEvent"])
        if train:
            self.train_patients = np.array(patients)
        else:
            self.test_patients = np.array(patients)

        return np.array(x), np.array(x2), np.array(x3), np.array(y)

    def setSeed(self):
        seed(self.opts["seed"])
        set_random_seed(self.opts["seed"])

    def loadModel(self, path, name):
        with open(path + "/" + name + "_split_model.json", 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(path + "/" + name + "_split_weights.hdf5")
        with open(path + "/" + name + "_split_onehotdict.json", 'r') as f:
            self.one_hot_dict = json.load(f)

    def intToEvent(self, value):
        one_hot = list(np.eye(len(self.one_hot_dict.keys()))[value])
        for k, v in self.one_hot_dict.items():
            if str(v) == str(one_hot):
                return k

    def predict_test(self, path, name):
        self.loadModel(path=path, name=name)
        y_prob = self.model.predict([self.X_test, self.X2_test, self.X3_test])
        y_pred = np.argmax(y_prob, axis=1)

        return y_pred, y_prob

    def predict(self, tss):
        """
        Predict from a list TimedStateSamples

        :param tss: list<TimedStateSamples>
        :return: tuple (DREAM-NAP output, translated next event)
        """
        if not isinstance(tss, list) or not isinstance(tss[0], TimedStateSample) :
            raise ValueError("Input is not a list with TimedStateSample")

        preds = []
        next_events = []
        for sample in tss:
            features = [list(itertools.chain(sample.export()["TimedStateSample"][0], sample.export()["TimedStateSample"][1], sample.export()["TimedStateSample"][2]))]
            features = self.stdScaler.transform(features)
            pred = np.argmax(self.model.predict(features), axis=1)
            preds.append(pred[0])
            for p in pred:
                next_events.append(self.intToEvent(p))
        return preds, next_events


    """ Callback """
    class EvaluationCallback(Callback):
        def __init__(self, X_test, X2_test, X3_test, Y_test, X_val, X2_val, X3_val, Y_val):
            self.X_test = X_test
            self.X2_test = X2_test
            self.X3_test = X3_test
            self.Y_test = Y_test
            self.Y_test_int = np.argmax(self.Y_test, axis=1)

            self.X_val = X_val
            self.X2_val = X2_val
            self.X3_val = X3_val
            self.Y_val = Y_val
            self.Y_val_int = np.argmax(self.Y_val, axis=1)

            self.test_accs = []
            self.losses = []

        def on_train_begin(self, logs={}):
            self.test_accs = []
            self.losses = []

        def on_epoch_end(self, epoch, logs={}):
            y_pred = self.model.predict([self.X_test, self.X2_test, self.X3_test])
            y_score = y_pred
            y_pred = y_pred.argmax(axis=1)

            test_acc = accuracy_score(self.Y_test_int, y_pred, normalize=True)
            test_loss, _ = self.model.evaluate([self.X_test, self.X2_test, self.X3_test], self.Y_test)

            precision, recall, fscore, _ = precision_recall_fscore_support(self.Y_test_int, y_pred, average='weighted', pos_label=None)
            auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="weighted")

            logs['test_acc'] = test_acc
            logs['test_prec_weighted'] = precision
            logs['test_rec_weighted'] = recall
            logs['test_loss'] = test_loss
            logs['test_fscore_weighted'] = fscore
            logs['test_auc_weighted'] = auc

            precision, recall, fscore, support = precision_recall_fscore_support(self.Y_test_int, y_pred, average='macro', pos_label=None)

            auc = multiclass_roc_auc_score(self.Y_test_int, y_pred, average="macro")
            logs['test_prec_mean'] = precision
            logs['test_rec_mean'] = recall
            logs['test_fscore_mean'] = fscore
            logs['test_auc_mean'] = auc

            y_val_score = self.model.predict([self.X_val, self.X2_val, self.X3_val])
            y_val_pred = y_val_score.argmax(axis=1)
            auc = multiclass_roc_auc_score(self.Y_val_int, y_val_pred, average="macro")
            logs['val_auc'] = auc