
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input, GRU
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,GlobalAveragePooling1D,Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from JoinAttLayer import Attention
from keras.optimizers import *

max_features = 180000
maxlen = 250

def clean_text( text ):
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+\-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    #
    return text

import pickle
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
test_y = pickle.load(open('../features/pool_gru_fasttext_adj1_10_feat.pkl'))[1]
test_y = pd.read_csv('../features/weighted_avg_sub.csv').iloc[:,1:].values
print(test_y.shape)

list_sentences_train = train["comment_text"].fillna("CVxTz").apply(clean_text).values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").apply(clean_text).values
print(y.shape)


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

print(X_train.shape,X_test.shape)

# check word_index
tmp_cnt = 0
for k in tokenizer.word_index:
    print(k,tokenizer.word_index[k])
    tmp_cnt += 1
    if tmp_cnt >5:
        break
word_idx = tokenizer.word_index

# read word2vec
#
word_vec_dict = {}
with open('./crawl-300d-2M.vec') as f:
    first_line_flag = True
    for line in f:
        if first_line_flag:
            first_line_flag= False
            continue
        v_list = line.rstrip().split(' ')
        k = str(v_list[0])
        v = np.array([float(x) for x in v_list[1:]])
        word_vec_dict[k] = v
print(len(word_vec_dict))
print('Preparing embedding matrix')

EMBEDDING_DIM = 300
nb_words = min(max_features,len(word_idx))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word,i in word_idx.items():
    if i >= max_features:
        continue
    else:
        if word in word_vec_dict:
            embedding_matrix[i] = word_vec_dict[word]
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
del word_vec_dict

from sklearn.metrics import log_loss,accuracy_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU,CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
import gc


def eval_val(y,train_x):
    res = 0
    acc_res = 0
    for i in range(6):
        curr_loss = log_loss(y[:,i],train_x[:,i])
        acc = accuracy_score(y[:,i],train_x[:,i].round())
        print(i,curr_loss,acc)
        res += curr_loss
        acc_res += acc
    print('final',res/6, acc_res/6)

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    att = Attention(maxlen)(x)
    conc = concatenate([att,avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
print('def model done')

from sklearn.model_selection import KFold


def kf_train(fold_cnt=3, rnd=1):
    kf = KFold(n_splits=fold_cnt, shuffle=False, random_state=233 * rnd)
    train_pred, test_pred = np.zeros((159571, 6)), np.zeros((153164, 6))
    for train_index, test_index in kf.split(X_train):
        # x,y
        curr_x, curr_y = X_train[train_index], y[train_index]
        hold_out_x, hold_out_y = X_train[test_index], y[test_index]

        # model
        model = get_model()
        batch_size = 64
        epochs = 8
        file_path = "weights_base.best.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(X_test,test_y,
                  batch_size=batch_size, epochs=1,
                  validation_data=(hold_out_x, hold_out_y),
                  callbacks=callbacks_list)
        # train and pred
        model.fit(curr_x, curr_y,
                  batch_size=batch_size, epochs=epochs,
                  validation_data=(hold_out_x, hold_out_y),
                  callbacks=callbacks_list)

        model.load_weights(file_path)
        #curr_x = np.concatenate([curr_x,X_test])
        #curr_y = np.concatenate([curr_y,test_y])
        
        y_test = model.predict(X_test)
        test_pred += y_test
        hold_out_pred = model.predict(hold_out_x)
        train_pred[test_index] = hold_out_pred

        # clear
        del model
        gc.collect()
        K.clear_session()
    test_pred = test_pred / fold_cnt
    print('-------------------------------')
    print('all eval', eval_val(y, train_pred))
    return train_pred, test_pred


print('def done')

import pickle
sample_submission = pd.read_csv("../input/sample_submission.csv")

train_pred,test_pred = kf_train(fold_cnt=10,rnd=4)
print(train_pred.shape,test_pred.shape)
sample_submission[list_classes] = test_pred
sample_submission.to_csv("../pool_gru1_fasttext_adj1_sample_10_diedai.gz", index=False, compression='gzip')
with open('../features/pool_gru_fasttext_adj1_10_diedai_feat.pkl','wb') as fout:
    pickle.dump([train_pred,test_pred],fout)
print(sample_submission.head())
print('===================================')
