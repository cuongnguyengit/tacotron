##### Get modules for dataloader
import codecs
import math
import numpy as np
import os
import re
import tensorflow as tf
import tensorflow.keras as keras
import unicodedata

##### Import .py 
from hyperparams import Hyperparams as hp
from utils import *
from kor_text import *

##### Define Dataloader
class DataLoader:

    def __init__(self, hp, maxlen = 200):

        self.batch_size = hp.batch_size
        self.maxlen = maxlen
        self.char2idx, self.idx2char = self.load_vocab()
        # Open different transcript based on type of source
        if hp.source == "vlsp2020":
            fpaths = []
            text_lengths = []
            texts = []
            for r, d, f in os.walk(hp.data):
                for file in f:
                    try:
                        if file.endswith(".txt"):
                            print(os.path.join(r, file))
                            with open(os.path.join(r, file), 'r', encoding='utf-8') as rf:
                                text = rf.read().strip()
                                text = [self.char2idx[char] for char in text]
                                fpath = os.path.join(r, file).replace('.txt', '.wav')
                                if not os.path.exists(fpath):
                                    continue
                            # Appending
                            fpaths.append(fpath)
                            text_lengths.append(len(text))
                            texts.append(np.array(text, np.int32).tostring())
                    except Exception as e:
                        print(file, e)
            self.fpaths, self.text_lengths, self.texts = fpaths, text_lengths, texts
            self.n_data = len(self.fpaths)

        else:
            transcript = os.path.join(hp.data, 'metadata.csv')
            self.lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            self.n_data = len(self.lines)
            self.fpaths, self.text_lengths, self.texts = self.load_data()  # list

        ##### Get total number of batches
        self.total_batch_num = len(self.fpaths) // self.batch_size

        ##### Sort based on text length
        self.fpath_text_dict = {l: self.texts[i] for i, l in enumerate(self.fpaths)}
        self.fpaths = self._sort_by_others(self.fpaths, self.text_lengths)
        self.texts = [self.fpath_text_dict[f] for f in self.fpaths]  # Done in this way to avoid wrong sortings among same length elements

        ##### Then, convert to tensor
        self.text_lengths = tf.convert_to_tensor(self.text_lengths)
        self.fpaths = tf.convert_to_tensor(self.fpaths)
        self.texts = tf.convert_to_tensor(self.texts)

        ##### Create dataloader
        concated_tensors = tf.concat([tf.expand_dims(self.fpaths, 1),
                                      tf.expand_dims(self.texts, 1)], axis=1)

        self.loader = tf.data.Dataset.from_tensor_slices(concated_tensors).\
            map(lambda x: self._mapping(x)).\
            padded_batch(self.batch_size, padded_shapes=([None], [None, None], [None, None]))


    ##### Function for ordering
    def _sort_by_others(self, target, key):

        """
        INPUT
        - target: numpy array of lements to be sorted
        - key: numpy array of text length

        OUTPUT
        - LIST
        """

        target_sorted = [x for _, x in sorted(zip(key, target))]

        return target_sorted
    

    ##### Function for padding
    def _padding(self, list_of_tensors, maxlen = None):
    
        if maxlen is None:
            maxlen = max([len(l_i) for l_i in list_of_tensors])
        
        # 1-dimensional padding
        if len(list_of_tensors[0].shape) == 1:
            padded_tensors = [tf.concat([t,tf.zeros(maxlen - len(t), dtype= tf.int32)], axis = 0) for t in list_of_tensors]
        # 2-dimensional padding
        else:
            padded_tensors = [tf.pad(t, tf.constant([[0, maxlen - len(t),],
                                                    [0, 0]])) for t in list_of_tensors]
        
        return tf.convert_to_tensor(padded_tensors)


    ##### Function to get dictionaries for indexing characters
    # @staticmethod
    def load_vocab(self):
        char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
        idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
        return char2idx, idx2char

    ##### Function for text normalizing
    # @staticmethod
    def text_normalize(self, text):
        # text = ''.join(char for char in unicodedata.normalize('NFD', text)
        #                     if unicodedata.category(char) != 'Mn') # Strip accents

        text = text.lower()
        text = re.sub("[^{}]".format(hp.vocab), " ", text)
        text = re.sub("[ ]+", " ", text)
        return text

    ##### Function for preprocessing 
    def load_data(self):
        fpaths, text_lengths, texts = [], [], []

        for i, line in enumerate(self.lines):

            # Case when using English
            if hp.source == "LJSpeech":
                fname, _, text = line.strip().split("|")
                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                text = self.text_normalize(text) + "E"  # E: EOS; end of the sentence
            # Case when using Vietnamese
            else:
                fname, _, text = line.strip().split("|")
                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                text = self.text_normalize(text)
                # text = cleaners.korean_cleaners(text)

            if not os.path.exists(fpath):
                print(fpath)
                continue

            text = [self.char2idx[char] for char in text]

            # Appending
            fpaths.append(fpath)
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())     
        
        return fpaths, text_lengths, texts

    ##### Function for decoding bytes to integers (int32)
    def _decode_map(self, text):
        decoded = tf.io.decode_raw(text, out_type=tf.int32)
        return decoded

    ##### Function for getting mel and linear(mag) spectrograms
    def _spectrogram_map(self, fpath):
        
        fname, mel, mag = tf.numpy_function(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        fname.set_shape(())
        mel.set_shape((None, hp.n_mels*hp.r))
        mag.set_shape((None, hp.n_fft//2+1))
        
        return fname, mel, mag

    ##### Function for mapping form filename string to actual spectrograms
    def _mapping(self, inputs):
        
        fpath = inputs[0]
        text = inputs[1]

        text_decoded = self._decode_map(text)
        fname, mel, mag = self._spectrogram_map(fpath)
        
        return text_decoded, mel, mag

# if __name__ == '__main__':
#     print(DataLoader.text_normalize('cho tôi hỏi'))
