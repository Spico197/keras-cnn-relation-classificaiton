from config import config


def load_word2vec():
    import numpy as np
    from utils import load_data

    words = set()
    train_data = load_data(config.TRAIN_DATA_PATH)
    test_data = load_data(config.TEST_DATA_PATH)
    for item in train_data:
        for token in item['sen_list']:
            if token.lower() not in words:
                words.add(token.lower())
    for item in test_data:
        for token in item['sen_list']:
            if token.lower() not in words:
                words.add(token.lower())

    wordvec_index = {}
    word_vectors = []

    with open(config.WORD2VEC_PATH, 'r', encoding="utf8") as file:
        i = 1
        for line in file:
            print("Process: {}".format(i), end="\r")
            i += 1
            split = line.strip().split()
            word = split[0]
            if len(wordvec_index) == 0:
                wordvec_index['PADDING_TOKEN'] = len(word_vectors)
                vector = np.zeros(len(split)-1)
                word_vectors.append(vector)

                wordvec_index['UNKNOWN_TOKEN'] = len(word_vectors)
                vector = np.random.normal(0, np.sqrt(0.25), len(split)-1)
                word_vectors.append(vector)
            if word.lower() in words:
                wordvec_index[word.lower()] = len(wordvec_index)
                try:
                    vector = np.array([float(x) for x in split[1:]])
                    word_vectors.append(vector)
                except ValueError as e:
                    # print(e)
                    # print(split)
                    print(split)
                    # raise ValueError
        print()
    # word_vectors = np.asarray(word_vectors).reshape((-1, config.EMBEDDING_DIM))
    word_vectors = np.array(word_vectors)
    # word_vectors = np.concatenate(word_vectors, axis=0)
    # word_vectors = word_vectors.reshape((-1, config.EMBEDDING_DIM))
    print("words length: {}".format(len(words)))
    print("wordvec_index length: {}".format(len(wordvec_index)))
    print("word_vectors.shape: {}".format(word_vectors.shape))
    return  wordvec_index, word_vectors

def get_vector_by_word(word, wordvec_index, word_vectors):
    if word.lower() in wordvec_index:
        return word_vectors[wordvec_index[word]]
    else:
        return word_vectors[wordvec_index["UNKNOWN_TOKEN"]]

def get_vectorid_by_word(word, wordvec_index):
    if word in wordvec_index:
        return wordvec_index[word]
    elif word.lower() in wordvec_index:
        return wordvec_index[word.lower()]
    return wordvec_index["UNKNOWN_TOKEN"]
