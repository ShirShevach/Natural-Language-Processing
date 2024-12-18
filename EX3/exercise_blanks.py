import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
NEGATE = "negate"
RARE = "rare"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    vec = np.zeros(embedding_dim)
    for word in sent.text:
        if word in word_to_vec:
            vec += word_to_vec[word]
    vec /= len(sent.text)
    return vec


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    V = len(word_to_ind)
    vec = np.zeros(V)
    for word in sent.text:
        ind = word_to_ind[word]
        one_hot = get_one_hot(V, ind)
        vec += one_hot
    vec /= len(sent.text)
    return vec

def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: i for i, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    embedding_matrix = np.zeros((seq_len, embedding_dim))
    sent_len = min(len(sent.text), seq_len)
    for i in range(sent_len):
        word = sent.text[i]
        if word in word_to_vec:
            embedding_matrix[i] = word_to_vec[word]
    return embedding_matrix


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=64,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # make negate and rare words examples:
        self.sentences[NEGATE] = []
        self.sentences[RARE] = []
        negate_ind = data_loader.get_negated_polarity_examples(self.sentences[TEST])
        self.sentences[NEGATE] = [self.sentences[TEST][i] for i in negate_ind]
        rare_ind = data_loader.get_rare_words_examples(self.sentences[TEST], self.sentiment_dataset)
        self.sentences[RARE] = [self.sentences[TEST][i] for i in rare_ind]

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=self.n_layers, dropout=self.dropout,
                            bidirectional=True, batch_first=True)

        # Linear layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Multiply by 2 for bidirectional LSTM

    def forward(self, text):
        # LSTM layer
        lstm_out, _ = self.lstm(text)

        # Concatenate the last hidden state from both directions
        combined = torch.cat((lstm_out[:, 0, self.hidden_dim:], lstm_out[:, -1, :self.hidden_dim]), dim=1)

        # Linear layer
        output = self.fc(combined)

        return output.squeeze(1)  # Squeeze to make it a 1D tensor

    def predict(self, text):
        pred_probs = self.forward(text)
        output = torch.sigmoid(pred_probs)
        # outoput = output.squeeze(1)  # Squeeze to make it a 1D tensor
        return output


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return self.sigmoid(self.forward(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    # Round predictions to the nearest integer (0 or 1)
    rounded_preds = torch.round(preds)

    # Compare predictions to labels
    correct = (rounded_preds == y).float()

    # Accuracy is number of correct predictions divided by the number of examples
    accuracy = correct.sum() / len(correct)
    return accuracy.item()  # Return accuracy as a Python scalar


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()
    sum_loss = sum_accuracy = num_batches = 0
    for data, y_true in tqdm(data_iterator):
        y_pred = model.forward(data.type(torch.FloatTensor)).flatten()
        y_pred_binary = model.predict(data.type(torch.FloatTensor)).flatten()
        loss = criterion(y_pred, y_true)

        sum_loss += loss.item()
        sum_accuracy += binary_accuracy(y_pred_binary, y_true)
        num_batches += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum_loss / num_batches, sum_accuracy / num_batches


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    sum_loss = sum_accuracy = num_batches = 0
    with torch.no_grad():
        for data, y_true in tqdm(data_iterator):
            y_pred = model.forward(data.type(torch.FloatTensor)).flatten()
            y_pred_binary = model.predict(data.type(torch.FloatTensor)).flatten()
            loss = criterion(y_pred, y_true)

            sum_loss += loss.item()
            sum_accuracy += binary_accuracy(y_pred_binary, y_true)
            num_batches += 1

    return sum_loss / num_batches, sum_accuracy / num_batches


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    criterion = nn.BCEWithLogitsLoss()
    test_set = data_iter.get_torch_iterator(data_subset=TEST)
    test_loss, test_accuracy = evaluate(model, test_set, criterion)
    print(f"Test set loss: {test_loss}")
    print(f"Test set accuracy: {test_accuracy}")

    test_negate_set = data_iter.get_torch_iterator(data_subset=NEGATE)
    test_loss, test_accuracy = evaluate(model, test_negate_set, criterion)
    print(f"Test negate set loss: {test_loss}")
    print(f"Test negate set accuracy: {test_accuracy}")

    test_rare_set = data_iter.get_torch_iterator(data_subset=RARE)
    test_loss, test_accuracy = evaluate(model, test_rare_set, criterion)
    print(f"Test rare set loss: {test_loss}")
    print(f"Test rare set accuracy: {test_accuracy}")


def train_model(name, model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    train_set = data_manager.get_torch_iterator(data_subset=TRAIN)
    val_set = data_manager.get_torch_iterator(data_subset=VAL)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loss_train = []
    accuracy_train = []
    loss_validation = []
    accuracy_validation = []
    for _ in range(n_epochs):
        loss, accuracy = train_epoch(model, train_set, optimizer, criterion)
        loss_train.append(loss)
        accuracy_train.append(accuracy)

        loss, accuracy = evaluate(model, val_set, criterion)
        loss_validation.append(loss)
        accuracy_validation.append(accuracy)

    plot_graph(name, loss_train, accuracy_train, loss_validation, accuracy_validation, n_epochs)
    get_predictions_for_data(model, data_manager)


def plot_graph(name, loss_train, accuracy_train, loss_validation, accuracy_validation, n_epochs):
    # Plot train and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), loss_train, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), loss_validation, label='Validation Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(f'{name}: Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot train and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), accuracy_train, label='Train Accuracy')
    plt.plot(range(1, n_epochs + 1), accuracy_validation, label='Validation Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title(f'{name}: Train and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager()
    V = len(data_manager.sentiment_dataset.get_word_counts().keys())
    log_linear_one_hot = LogLinear(V)
    train_model("Log Linear with one hot", log_linear_one_hot, data_manager, 20, 0.01, 0.001)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, embedding_dim=W2V_EMBEDDING_DIM)
    log_linear_w2v = LogLinear(W2V_EMBEDDING_DIM)
    train_model("Log Linear with w2v", log_linear_w2v, data_manager, 20, 0.01, 0.001)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, embedding_dim=W2V_EMBEDDING_DIM)
    lstm = LSTM(W2V_EMBEDDING_DIM, 100, 1, 0.5)
    train_model("lstm", lstm, data_manager, 4, 0.001, 0.0001)


if __name__ == '__main__':
    print("Log Linear with one hot:")
    train_log_linear_with_one_hot()
    print("\nLog Linear with w2v:")
    train_log_linear_with_w2v()
    print("\nLSTM:")
    train_lstm_with_w2v()