# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
import nltk


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, inp, hidden, embeddings, training_bool = False, train_return_bool = False):

        # initialize WordEmbedding object to use in predict
        self.embeddings = embeddings
        self.training_bool = training_bool
        self.hidden = hidden
        self.input = inp
        self.train_return_bool = train_return_bool

        # needs a nn.Module to act as the model
        class NN(nn.Module):
            # change the input to account for a batch of data points
            def __init__(self, input_size, hidden_size, out_size=2, training_bool = self.training_bool, embeddings=self.embeddings):
                super(NN, self).__init__()

                # add embedding layer before computation
                # forward will take in a tensor of word indices
                # the embedding layer will convert these indices into embeddings
                if training_bool:
                    self.embedding_layer = embeddings.get_initialized_embedding_layer(frozen = True)
                else:
                    self.embedding_layer = embeddings.get_initialized_embedding_layer(frozen = True)
                
                # create the NN structure
                self.V = nn.Linear(input_size, hidden_size)
                # self.g = nn.Tanh()
                self.g = nn.ReLU()
                self.W = nn.Linear(hidden_size, out_size)
                self.softmax = nn.Softmax(dim = 0)
                # batching softmax
                #self.softmax = nn.Softmax(dim=1)
                
                # initialize the weights
                nn.init.xavier_uniform_(self.V.weight)
                nn.init.xavier_uniform_(self.W.weight)
                # Initialize with zeros instead
                # nn.init.zeros_(self.V.weight)
                # nn.init.zeros_(self.W.weight)

            def forward(self, x):
                return self.softmax(self.W(self.g(self.V(torch.mean(self.embedding_layer(x), dim=0)))))
                
                # return statement for batching
                return self.softmax(self.W(self.g(self.V(torch.mean(self.embedding_layer(x), dim=1)))))
            
        self.DAT = NN(input_size=self.input, hidden_size=self.hidden)
    
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        # ex_words is a sentence 
        # e.g. ex_words = ["This", "is", "a", "sentence"]

        # need to convert words to index and then find embeddings
        # 1) convert words to index
        # create numpy array of embeddings
        # embeddings should be dim len(ex_words) x self.embeddings.get_embeddings_length
        # handle indices of -1 by replacing with indicator for "UNK" (zero vector)
        # only do this step on dev set evaluation and not training set evaluation

        punctuation = [",", ".", ";", ":", "/", "-", "--", "!", "(", ")", "'", '"']
        ex_words = [word for word in ex_words if word not in punctuation]

        if has_typos==True and self.training_bool==False:
            vector_of_indices = []
            corrected_words = {}
            
            for word in ex_words:
                if '-' in word or "'" in word:
                    word = "UNK"
                index = self.embeddings.word_indexer.index_of(word)
                if index == -1 and len(word) > 3:
                    # check if already in corrected_words:
                    if word in corrected_words.keys():
                        correct_word = corrected_words[word]
                        corrected_index = self.embeddings.word_indexer.index_of(correct_word)
                        vector_of_indices.append(corrected_index)
                    else:
                        # correct the word using nltk.edit_distance
                        # only need to look at words in the indexer that have length greater than 3
                        edit_dist = []
                        indices_ed = []
                        ##for idx in range(len(self.embeddings.word_indexer)):
                        possible_keys = [(key, idx) for key, idx in self.embeddings.word_indexer.objs_to_ints.items() if len(key) > 3 and word[0:3] == key[0:3]]
                        for idx_word, idx in possible_keys:
                            ##word_in_indexer = self.embeddings.word_indexer.get_object(idx)
                            # check only words with greater than length 3
                            ##if len(word_in_indexer) > 3 and word[0:3] == word_in_indexer[0:3]:
                            ##if word[0:3] == idx_word[0:3]:
                            edit_dist.append(nltk.edit_distance(word, idx_word))
                            indices_ed.append(idx)
                        if len(edit_dist) == 0:
                            vector_of_indices.append(self.embeddings.word_indexer.index_of("UNK"))
                        else:
                            index_of_max = edit_dist.index(min(edit_dist))
                            corrected_index = indices_ed[index_of_max]
                            corrected_words[word] = self.embeddings.word_indexer.get_object(corrected_index)
                            vector_of_indices.append(corrected_index)
                        
                elif index == -1 and len(word) <= 3:
                    vector_of_indices.append(self.embeddings.word_indexer.index_of("UNK"))
                else:
                    vector_of_indices.append(index)
            vector_of_indices = np.array(vector_of_indices)
        else:
            vector_of_indices = np.array([self.embeddings.word_indexer.index_of(word) if self.embeddings.word_indexer.index_of(word) != -1 else self.embeddings.word_indexer.index_of("UNK") for word in ex_words])   

        # 2) convert numpy array of embeddings to a tensor
        tensor_of_indices = torch.from_numpy(vector_of_indices)

        # 'log_probs' is a tensor with probabilities
        # pass in the tensor of embeddings
        log_probs = self.DAT(tensor_of_indices)

        # return the argmax
        # need to take the higher of the probabilities
        if self.train_return_bool:
            return torch.argmax(log_probs), log_probs
        else:
            return torch.argmax(log_probs)
    
    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:

        # solution without batching
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]
    
        # solution with batching
        for sentence in all_ex_words:
            self.embeddings.word_indexer.index_of(word) 


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    # import args
    # args.num_epochs, args.lr
     
    # SentimenExample?

    # initialize the Classifier object
    classifier = NeuralSentimentClassifier(inp=word_embeddings.get_embedding_length(), hidden=4, embeddings=word_embeddings, train_return_bool=True, training_bool=True)
    # get the DAT NN
    dat_nn = classifier.DAT

    # initialize optimizer
    optimizer = optim.Adam(dat_nn.parameters(), lr = args.lr) # hard code here

    # iterate through each epoch
    for epoch in range(args.num_epochs):
        # iterate through the training data
        # randomize training data
        total_loss = 0
        loss_fnc = nn.CrossEntropyLoss()
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        for i in ex_indices:
            train_ex = train_exs[i]
            X = train_ex.words

            # one hot encode labels
            y = train_ex.label
            y_onehot = torch.zeros(2)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)

            # zero out gradient
            dat_nn.zero_grad()
            # pass through network 
            if train_model_for_typo_setting:
                y_pred, log_probs = classifier.predict(ex_words = X, has_typos = True)
            else:
                y_pred, log_probs = classifier.predict(ex_words = X, has_typos = False)

            # calculate the loss
            # can alternatively use other loss functions
            loss = loss_fnc(log_probs, y_onehot)
            #loss = torch.neg(log_probs).dot(y_onehot)
            
            # loss for batching
            # loss = torch.sum(loss_fnc(log_probs, y_onehot))
            total_loss += loss
            
            # compute backwards gradient
            loss.backward()
            # step with the optimizer
            optimizer.step()
        
        # print out loss for epoch to assess performance when training
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    classifier.train_return_bool = False
    classifier.training_bool = False
    return classifier
