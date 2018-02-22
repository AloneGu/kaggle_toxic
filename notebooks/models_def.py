from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,GlobalAveragePooling1D,Conv1D,Conv2D,Reshape
from keras.preprocessing import text, sequence
from keras.layers import MaxPool2D,concatenate,Flatten,CuDNNGRU,GRU,MaxPooling1D

def cnn2d(maxlen,nb_words,embed_dim,embedding_matrix,trainable_flag,comp=True):
    # https://github.com/bhaveshoswal/CNN-text-classification-keras/blob/master/model.py
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is None:
        x = Embedding(nb_words, embed_dim)(inp)
    else:
        x = Embedding(nb_words, embed_dim, weights=[embedding_matrix],trainable=trainable_flag)(inp)
    x = Reshape((maxlen,embed_dim,1))(x)
    x = Dropout(0.2)(x)
   
    x1 = Conv2D(128,kernel_size=(3,embed_dim),activation='relu')(x)
    x1 = MaxPool2D(pool_size=(maxlen - 3 + 1, 1), strides=(1,1), padding='valid')(x1)
    
    x2 = Conv2D(128,kernel_size=(5,embed_dim),activation='relu')(x)
    x2 = MaxPool2D(pool_size=(maxlen - 5 + 1, 1), strides=(1,1), padding='valid')(x2)
    
    x3 = Conv2D(128,kernel_size=(7,embed_dim),activation='relu')(x)
    x3 = MaxPool2D(pool_size=(maxlen - 7 + 1, 1), strides=(1,1), padding='valid')(x3)
    
    x = concatenate([x1,x2,x3])
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    if comp:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return model

def cnn_v1(maxlen,nb_words,embed_dim,embedding_matrix,trainable_flag,comp=True):
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is None:
        x = Embedding(nb_words, embed_dim)(inp)
    else:
        x = Embedding(nb_words, embed_dim, weights=[embedding_matrix],trainable=trainable_flag)(inp)
    x = Dropout(0.2)(x)
    x = Conv1D(256,
             3,
             padding='valid',
             activation='relu',
             strides=1)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    if comp:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return model

def cnn_v2(maxlen,nb_words,embed_dim,embedding_matrix,trainable_flag,comp=True):
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is None:
        x = Embedding(nb_words, embed_dim)(inp)
    else:
        x = Embedding(nb_words, embed_dim, weights=[embedding_matrix],trainable=trainable_flag)(inp)
    x = Conv1D(384,
             5,
             padding='valid',
             activation='relu',
             strides=1)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    if comp:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return model

def cnn_gru(maxlen,nb_words,embed_dim,embedding_matrix,trainable_flag,comp=True):
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is None:
        x = Embedding(nb_words, embed_dim)(inp)
    else:
        x = Embedding(nb_words, embed_dim, weights=[embedding_matrix],trainable=trainable_flag)(inp)
    x = Dropout(0.2)(x)
    main = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(64)(main)
    main = Dense(32, activation="relu")(main)
    main = Dense(6, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    if comp:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return model

def cudnn_gru(maxlen,nb_words,embed_dim,embedding_matrix,trainable_flag,comp=True):
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is None:
        x = Embedding(nb_words, embed_dim)(inp)
    else:
        x = Embedding(nb_words, embed_dim, weights=[embedding_matrix],trainable=trainable_flag)(inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=False))(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    if comp:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

def lstm_v1(maxlen,nb_words,embed_dim,embedding_matrix,trainable_flag,comp=True):
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is None:
        x = Embedding(nb_words, embed_dim)(inp)
    else:
        x = Embedding(nb_words, embed_dim, weights=[embedding_matrix],trainable=trainable_flag)(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    if comp:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

def gru_v1(maxlen,nb_words,embed_dim,embedding_matrix,trainable_flag,comp=True):
    inp = Input(shape=(maxlen, ))
    if embedding_matrix is None:
        x = Embedding(nb_words, embed_dim)(inp)
    else:
        x = Embedding(nb_words, embed_dim, weights=[embedding_matrix],trainable=trainable_flag)(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    if comp:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model