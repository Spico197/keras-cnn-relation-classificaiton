import os

class Config(object):
    def __init__(self, **kwargs):
        self.DATA_DIR = "./data/"
        self.TRAIN_DATA_PATH = os.path.join(self.DATA_DIR, "TRAIN_FILE.TXT")
        self.TEST_DATA_PATH = os.path.join(self.DATA_DIR, "TEST_FILE_FULL.TXT")
        self.MODEL_SAVE_PATH = "./models/keras.model"

        self.WIKI_WORD2VEC_MODEL_PATH = r"E:\GZU\GraduationDesign\src\ref\deeplearning4nlp-tutorial\2017-07_Seminar\Session 3 - Relation CNN\code\embeddings\wiki_extvec"
        self.GLOVE_MODEL_PATH = r"D:\NLP\glove.6B\glove.6B.300d.txt"
        self.WORD2VEC_MODEL = "wiki"
        self.WORD2VEC_PATH = self.WIKI_WORD2VEC_MODEL_PATH

        if kwargs.get('word2vec_model'):
            if kwargs['word2vec_model'] == "wiki":
                self.WORD2VEC_MODEL = "wiki"
                self.WORD2VEC_PATH = self.WIKI_WORD2VEC_MODEL_PATH
            elif kwargs['word2vec_model'] == "glove":
                self.WORD2VEC_MODEL = "glove"
                self.WORD2VEC_PATH = self.GLOVE_MODEL_PATH

        self.EMBEDDING_DIM = 300
        self.POSITION_DIM = 50
        self.FILTER_SIZE = 3
        self.FILTER_NUMBER = 100
        self.DROP_VAL = 0.25
        self.HIDDEN_LAYER1 = 200
        self.HIDDEN_LAYER2 = 100
        self.LEARNING_RATE = 0.001
        self.MAX_TOKEN_LENGTH = 97
        self.BATCH_SIZE = 64
        self.EPOCHS_NUMBER = 100

config = Config(word2vec_model='glove')
