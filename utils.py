import re
import numpy as np

from config import config
from word2vec import load_word2vec, get_vectorid_by_word


def load_data(filepath):
    regex = r"(\d+)\t\"(.*<e1>(.*)<\/e1>.*<e2>(.*)<\/e2>.*)\"(.*)\n(.*)\nComment:(.*)"
    data_string = ""
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        data_string = "".join(file.readlines())
    if data_string == "":
        raise ValueError("data string is empty!")
    
    matches = re.finditer(regex, data_string)
    for match in matches:
        sen_list = match.group(2)
        sen_list = sen_list.replace("'", " '")
        sen_list = sen_list.replace("<e1>", " <e1> ")
        sen_list = sen_list.replace("</e1>", " </e1> ")
        sen_list = sen_list.replace("<e2>", " <e2> ")
        sen_list = sen_list.replace("</e2>", " </e2> ")
        sen_list.strip()
        if config.WORD2VEC_MODEL == "google_news":
            for ch in '!"#$%&()*+,-.:;=?@[\]^_`{|}~':
                if ch in sen_list:
                    sen_list = sen_list.replace(ch, ' ')
        elif config.WORD2VEC_MODEL == "wiki":
            for ch in '.!"#$%&()*+,-:;=?@[\]^_`{|}~':
                if ch in sen_list:
                    sen_list = sen_list.replace(ch, " {} ".format(ch))
        sen_list = sen_list.split()

        e1_pos = sen_list.index('<e1>') + 1
        e2_pos = sen_list.index('<e2>') + 1
        e1_left = ""; e1_right = ""; e2_left = ""; e2_right = ""

        if e1_pos - 2 >= 0:
            e1_left = sen_list[e1_pos - 2]
        if e1_pos + 2 <= len(sen_list) - 1:
            if sen_list[e1_pos + 2] == '<e2>':
                e1_right = sen_list[e1_pos + 3]
            else:
                e1_right = sen_list[e1_pos + 2]
        if e2_pos - 2 >= 0:
            if sen_list[e2_pos - 2] == '</e1>':
                e2_left = sen_list[e2_pos - 3]
            else:
                e2_left = sen_list[e2_pos - 2]
        if e2_pos + 2 <= len(sen_list) - 1:
            e2_right = sen_list[e2_pos + 2]

        sen_list.remove("<e1>")
        sen_list.remove("</e1>")
        sen_list.remove("<e2>")
        sen_list.remove("</e2>")
        e1_pos -= 1; e2_pos -= 1

        data.append({"id": match.group(1).strip(), "sentence": " ".join(sen_list), "sen_list": sen_list,
            "e1": match.group(3).strip(), "e2": match.group(4).strip(), "e1_pos": e1_pos, "e2_pos": e2_pos,
            "e1_left": e1_left, "e1_right": e1_right, "e2_left": e2_left, "e2_right": e2_right,
            "relation": match.group(6).strip(), "Comment": match.group(7).strip()})

    return data

def preprocessing(filepath, wordvec_index):
    distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}

    data = load_data(filepath)
    labels = [x['relation'] for x in data]
    
    token_mat = []; pos_mat1 = []; pos_mat2 = []
    token_indexes = np.zeros((1, config.MAX_TOKEN_LENGTH))
    pos_val1 = np.zeros((1, config.MAX_TOKEN_LENGTH))
    pos_val2 = np.zeros((1, config.MAX_TOKEN_LENGTH))

    for item in data:
        token_indexes[0, :len(item['sen_list'])] = [get_vectorid_by_word(word, wordvec_index) for word in item['sen_list']]
        pos_val1[0, :config.MAX_TOKEN_LENGTH] = [i - int(item['e1_pos']) for i in range(0, config.MAX_TOKEN_LENGTH)]
        pos_val2[0, :config.MAX_TOKEN_LENGTH] = [i - int(item['e2_pos']) for i in range(0, config.MAX_TOKEN_LENGTH)]
    
        token_mat.append(token_indexes)
        pos_mat1.append(pos_val1)
        pos_mat2.append(pos_val2)

    return {"labels": np.array(labels), "token_mat": np.array(token_mat, dtype='int32'), 
            "pos_mat1": np.array(pos_mat1, dtype='int32'), "pos_mat2": np.array(pos_mat2, dtype='int32')}

def main():
    from sklearn.preprocessing import LabelEncoder
    import pickle

    print("-"*20 + " " + config.WORD2VEC_MODEL + " word2vec model is loading " + "-"*20)

    try:
        with open("preprocessed_data/wiki_word2vec.ext", 'rb') as file:
            word2vec = pickle.load(file)
            wordvec_index = word2vec['wordvec_index']
            word_vectors = word2vec['word_vectors']
    except:
        wordvec_index, word_vectors = load_word2vec()
        word2vec = {"wordvec_index": wordvec_index, "word_vectors": word_vectors}
        with open("preprocessed_data/wiki_word2vec", 'wb') as file:
            pickle.dump(word2vec, file)

    train_set = preprocessing(config.TRAIN_DATA_PATH, wordvec_index)
    test_set = preprocessing(config.TEST_DATA_PATH, wordvec_index)

    train_set['token_mat'] = train_set['token_mat'].reshape((train_set['token_mat'].shape[0], train_set['token_mat'].shape[2]))
    train_set['pos_mat1'] = train_set['pos_mat1'].reshape((train_set['pos_mat1'].shape[0], train_set['pos_mat1'].shape[2]))
    train_set['pos_mat2'] = train_set['pos_mat2'].reshape((train_set['pos_mat2'].shape[0], train_set['pos_mat2'].shape[2]))
    test_set['token_mat'] = test_set['token_mat'].reshape((test_set['token_mat'].shape[0], test_set['token_mat'].shape[2]))
    test_set['pos_mat1'] = test_set['pos_mat1'].reshape((test_set['pos_mat1'].shape[0], test_set['pos_mat1'].shape[2]))
    test_set['pos_mat2'] = test_set['pos_mat2'].reshape((test_set['pos_mat2'].shape[0], test_set['pos_mat2'].shape[2]))


    lbe = LabelEncoder()
    lbe.fit(train_set['labels'])
    lbe.fit(test_set['labels'])
    train_set['labels'] = np.array(lbe.transform(train_set['labels']), dtype='int32')
    test_set['labels'] = np.array(lbe.transform(test_set['labels']), dtype='int32')
    print("train:\n\ttoken_mat.shape: {}, pos_mat1.shape: {}, pos_mat2.shape: {}"
            .format(train_set['token_mat'].shape, train_set['pos_mat1'].shape, train_set['pos_mat2'].shape))
    print("test:\n\ttoken_mat.shape: {}, pos_mat1.shape: {}, pos_mat2.shape: {}"
            .format(test_set['token_mat'].shape, test_set['pos_mat1'].shape, test_set['pos_mat2'].shape))
    print("data in train/test: {}/{}".format(len(train_set['labels']), len(test_set['labels'])))
    print("There are {} classes in dataset".format(len(lbe.classes_)))
    
    data = {
        "wordvec_index": wordvec_index, "word_vectors": word_vectors,
        "train_set": train_set, "test_set": test_set, "label_encoder": lbe
    }
    
    with open('preprocessed_data/data.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    main()
