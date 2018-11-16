import pickle
import numpy as np

from config import config

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model


# with open("preprocessed_data/data.pkl", 'rb') as file:
#     data = pickle.load(file)
with open("preprocessed_data/sem-relations.pkl", 'rb') as file:
    data = pickle.load(file)

# embeddings = data['word_vectors']
embeddings = data['wordEmbeddings']
print("word_vectors.shape: {}".format(embeddings.shape))
# print(embeddings[:2])
# y_train, X_train, pos_train_mat1, pos_train_mat2 = data['train_set'].values()
# y_test, X_test, pos_test_mat1, pos_test_mat2 = data['test_set'].values()
y_train, X_train, pos_train_mat1, pos_train_mat2 = data['train_set']
y_test, X_test, pos_test_mat1, pos_test_mat2 = data['test_set']

# out_number = len(data['label_encoder'].classes_)
out_number = 19
# y_train = to_categorical(y_train, num_classes=out_number)
# y_test = to_categorical(y_test, num_classes=out_number)

token_input = Input(shape=(config.MAX_TOKEN_LENGTH,), name="token_input")
tokens = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(token_input)

distance1_input = Input(shape=(config.MAX_TOKEN_LENGTH,), name='distance1_input')
distance1 = Embedding(config.MAX_TOKEN_LENGTH, config.POSITION_DIM)(distance1_input)

distance2_input = Input(shape=(config.MAX_TOKEN_LENGTH,), name='distance2_input')
distance2 = Embedding(config.MAX_TOKEN_LENGTH, config.POSITION_DIM)(distance2_input)

output = concatenate([tokens, distance1, distance2])

output = Convolution1D(filters=config.FILTER_NUMBER,
                        kernel_size=config.FILTER_SIZE,
                        padding='same',
                        activation='tanh',
                        strides=1)(output)
output = GlobalMaxPooling1D()(output)
output = Dropout(config.DROP_VAL)(output)
output = Dense(config.HIDDEN_LAYER1, activation='tanh')(output)
output = Dropout(config.DROP_VAL)(output)
output = Dense(config.HIDDEN_LAYER2, activation='tanh')(output)
output = Dropout(config.DROP_VAL)(output)
output = Dense(out_number, activation='softmax')(output)


model = Model(inputs=[token_input, distance1_input, distance2_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam',
                metrics=['acc'])
model.summary()

# model.fit([X_train, pos_train_mat1, pos_train_mat2], y_train, 
#             batch_size=config.BATCH_SIZE, epochs=1, 
#             validation_data=([X_test, pos_test_mat1, pos_test_mat2], y_test),
#             callbacks=[TensorBoard(log_dir='./tmp/logs/cnn_train/')])

max_prec, max_rec, max_acc, max_f1 = 0,0,0,0
def getPrecision(pred_test, y_test, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == y_test[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

def predict_classes(prediction):
    return prediction.argmax(axis=-1)

f1s = []
macrof1s = []
for i in range(0, config.EPOCHS_NUMBER):
    print("Process: {}/{}".format(i+1, config.EPOCHS_NUMBER))
    model.fit([X_train, pos_train_mat1, pos_train_mat2], y_train, 
                batch_size=config.BATCH_SIZE, epochs=1,
                callbacks=[TensorBoard(log_dir='./tmp/logs/cnn_train/')])
    pred_test = predict_classes(model.predict([X_test, pos_test_mat1, pos_test_mat2], verbose=False))
    
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(y_test)
   
    acc =  np.sum(pred_test == y_test) / float(len(y_test))
    max_acc = max(max_acc, acc)
    print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

    f1Sum = 0
    f1Count = 0
    for targetLabel in range(1, max(y_test)):        
        prec = getPrecision(pred_test, y_test, targetLabel)
        recall = getPrecision(y_test, pred_test, targetLabel)
        f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
        f1s.append(f1)
        f1Sum += f1
        f1Count +=1    
        
        
    macroF1 = f1Sum / float(f1Count)    
    macrof1s.append(macroF1)
    max_f1 = max(max_f1, macroF1)
    print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))

with open("preprocessed_data/result.pkl", 'wb') as file:
    result = {
        "max_prec": max_prec, 
        "max_rec": max_rec, 
        "max_acc": max_acc, 
        "max_f1": max_f1,
        "f1s": f1s, 
        "macrof1s": macrof1s
    }
    pickle.dump(result, file)
plot_model(model, to_file="{}.png".format("CNN Relation Classification"), show_shapes=True)
model.save(config.MODEL_SAVE_PATH)
