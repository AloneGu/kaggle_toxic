from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,GlobalAveragePooling1D,Conv1D,Conv2D,Reshape
from keras.preprocessing import text, sequence
from keras.layers import MaxPool2D,concatenate,Flatten,CuDNNGRU,GRU,MaxPooling1D
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

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

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim