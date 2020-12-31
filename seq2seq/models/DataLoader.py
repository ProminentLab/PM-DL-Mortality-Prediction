import numpy as np


class DataLoader:
    def __init__(self, train_file, train_labels_file, train_admissions, test_file, test_labels_file, test_admissions, embeddings_file,
                 val_file, val_labels_file, val_admissions):
        self.loadEmbeddings(embeddings_file=embeddings_file)
        self.loadSequences(train_file=train_file, train_labels_file=train_labels_file, test_file = test_file, test_labels_file=test_labels_file, val_file = val_file, val_labels_file = val_labels_file,
                           train_admissions=train_admissions, test_admissions=test_admissions, val_admissions=val_admissions)


    def loadSequences(self, train_file, train_labels_file, test_file, test_labels_file, val_file, val_labels_file, train_admissions, test_admissions, val_admissions):
        sequences = []
        labels = []
        self.max_seq_len = 0
        with open(train_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                seq = line.split(" ")
                if len(seq) > self.max_seq_len:
                    self.max_seq_len = len(seq)
                sequences.append(seq)

        with open(train_labels_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                labels.append(int(line))
        self.labels = self.one_hot_encode(labels, n_unique=3)

        self.X_in = np.zeros((len(sequences), self.max_seq_len, self.dim))
        self.X_out = np.zeros((len(sequences), self.max_seq_len, self.dim))
        self.X_out_mask = np.zeros((len(sequences), self.max_seq_len, self.dim))

        print("Sequence shape", self.X_in.shape)
        print("Labels shape",self.labels.shape)

        for i in range(self.X_in.shape[0]):
            self.X_in[i, :, :], _ = self.seq2emb(sequences[i], pad="left")
            self.X_out[i, :, :], self.X_out_mask[i, :, :] = self.seq2emb(sequences[i], pad="right")

        #### TEST PART #####
        sequences = []
        test_labels = []
        with open(test_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                sequences.append(line.split(" "))

        self.test_X_in = np.zeros((len(sequences), self.max_seq_len, self.dim))
        self.test_X_out = np.zeros((len(sequences), self.max_seq_len, self.dim))
        self.test_X_out_mask = np.zeros((len(sequences), self.max_seq_len, self.dim))

        for i in range(self.test_X_in.shape[0]):
            self.test_X_in[i, :, :], _ = self.seq2emb(sequences[i], pad="left")
            self.test_X_out[i, :, :], self.test_X_out_mask[i, :, :] = self.seq2emb(sequences[i], pad="right")

        with open(test_labels_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                test_labels.append(int(line))
        self.test_labels = self.one_hot_encode(test_labels, n_unique=3)


        #### VAL PART #####
        sequences = []
        val_labels = []
        with open(val_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                sequences.append(line.split(" "))

        self.val_X_in = np.zeros((len(sequences), self.max_seq_len, self.dim))
        self.val_X_out = np.zeros((len(sequences), self.max_seq_len, self.dim))
        self.val_X_out_mask = np.zeros((len(sequences), self.max_seq_len, self.dim))

        for i in range(self.val_X_in.shape[0]):
            self.val_X_in[i, :, :], _ = self.seq2emb(sequences[i], pad="left")
            self.val_X_out[i, :, :], self.val_X_out_mask[i, :, :] = self.seq2emb(sequences[i], pad="right")

        with open(val_labels_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                val_labels.append(int(line))
        self.val_labels = self.one_hot_encode(val_labels, n_unique=3)


        self.train_admissions = []
        self.test_admissions = []
        self.val_admissions = []
        with open(train_admissions, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.replace("\n", "")
                self.train_admissions.append(line)

        with open(test_admissions, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.replace("\n", "")
                self.test_admissions.append(line)

        with open(val_admissions, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.replace("\n", "")
                self.val_admissions.append(line)

    def seq2emb(self, sequence, pad="right"):
        seq_pattern = np.zeros((self.max_seq_len, self.dim))
        mask_pattern = np.zeros((self.max_seq_len, self.dim))
        seq = []
        for word in sequence:
            if word in self.word2idx.keys():
                seq.append(self.embeddings[self.word2idx[word]])
            else:
                seq.append(self.embeddings[self.word2idx["<unk>"]])

        if pad == "left":
            varcount = len(seq)
            for i in range(varcount):
                pos = self.max_seq_len - varcount + i
                if pos < 0:
                    continue
                seq_pattern[pos, ] = seq[i]
                mask_pattern[pos,] = np.ones(shape=(len(seq[i])))

        elif pad == "right":
            varcount = len(seq)
            for i in range(varcount):
                seq_pattern[i, ] = seq[i]
                mask_pattern[i,] = np.ones(shape=(len(seq[i])))

        return seq_pattern, mask_pattern


    def loadEmbeddings(self, embeddings_file):
        self.word2idx = {}  # dictionary of words to ids
        self.idx2word = {}  # dictionary of ids to words
        self.embeddings = []  # the word embeddings matrix

        self.dim = None
        with open(embeddings_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i >= 2:
                    self.dim = len(line.split(" ")) - 2
                    break

        with open(embeddings_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                # skip the first row if it is a header
                if i == 1:
                    if len(line.split()) < self.dim:
                        header = True
                        continue

                line = line.replace("\n", "")
                values = line.split(" ")
                word = values[0]
                self.word2idx[word] = i-2
                self.idx2word[i - 2] = word

                vector = np.asarray(values[1:-1], dtype='float32')
                self.embeddings.append(vector)

            # add an unk token, for OOV words
            # if "<unk>" not in word2idx:
            self.idx2word[len(self.idx2word) + 1] = "<unk>"
            self.word2idx["<unk>"] = len(self.word2idx)
            self.embeddings.append(np.random.uniform(low=-0.05, high=0.05, size=self.dim))

            print(set([len(x) for x in self.embeddings]))

            print('Found %s word vectors.' % len(self.embeddings))

            self.embeddings = np.array(self.embeddings, dtype='float32')

    def one_hot_encode(self, labels, n_unique):
        encoding = list()
        for value in labels:
            vector = [0 for _ in range(n_unique)]
            vector[value] = 1
            encoding.append(vector)
        return np.array(encoding)