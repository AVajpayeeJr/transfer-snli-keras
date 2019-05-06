from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Bidirectional, Concatenate, Dense, Dropout, Embedding, Input, LSTM
from keras.models import Model
import numpy as np


__author__ = 'avijitv'


class TemporalMaxPooling(Layer):
    def __init__(self, **kwargs):
        super(TemporalMaxPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, x, mask=None):
        if mask is None:
            mask = K.sum(K.ones_like(x), axis=-1)

        # if masked, set to large negative value so we ignore it when taking max of the sequence
        # K.switch with tensorflow backend is less useful than Theano's
        if K._BACKEND == 'tensorflow':
            mask = K.expand_dims(mask, axis=-1)
            mask = K.tile(mask, (1, 1, K.int_shape(x)[2]))
            masked_data = K.tf.where(K.equal(mask, K.zeros_like(mask)),
                K.ones_like(x)*-np.inf, x)  # if masked assume value is -inf
            return K.max(masked_data, axis=1)
        else:  # theano backend
            mask = mask.dimshuffle(0, 1, "x")
            masked_data = K.switch(K.eq(mask, 0), -np.inf, x)
            return masked_data.max(axis=1)

    def compute_mask(self, input, mask):
        # do not pass the mask to the next layers
        return None


class MaxPoolBiLSTM:
    def __init__(self, config, vocab_size, embedding_matrix, max_seq_len, name='encoder'):
        self.embedding_dim = config['word_embedding_dim']
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.max_seq_len = max_seq_len
        self.lstm_unit_dim = config['BiLSTM-Max']['lstm_unit_dim']
        self.encoding_dropout = config['encoding_dropout']
        self.name = name

    def get_model(self):
        seq_input = Input(shape=(self.max_seq_len,), dtype='int32', name=self.name + '_input')

        word_embedding = Embedding(output_dim=self.embedding_dim,
                                   input_dim=self.vocab_size,
                                   weights=[self.embedding_matrix],
                                   mask_zero=True,
                                   input_length=self.max_seq_len,
                                   trainable=False)(seq_input)
        bilstm = Bidirectional(LSTM(units=self.lstm_unit_dim,
                                    return_sequences=True))(word_embedding)

        encoding = TemporalMaxPooling()(bilstm)
        encoding = Dropout(self.encoding_dropout, name=self.name)(encoding)
        return encoding, seq_input

# TODO: Add support for other encoder architectures (LSTM, GRU, BiGRU, BiLSTM-Mean, Self-Attentive, Hier-Conv)
# TODO: Add support for non-linear final classification and multi-hidden layers post encoding
class NLIClassifier:
    def __init__(self, config, vocab_size, embedding_matrix, max_seq_len):
        if config['encoder']['type'] == 'BiLSTM-Max':
            self.premise_encoder, self.premise_input = MaxPoolBiLSTM(config=config['encoder'],
                                                                     vocab_size=vocab_size,
                                                                     embedding_matrix=embedding_matrix,
                                                                     max_seq_len=max_seq_len,
                                                                     name='premise_encoder').get_model()
            self.hypothesis_encoder, self.hypothesis_input = MaxPoolBiLSTM(config=config['encoder'],
                                                                           vocab_size=vocab_size,
                                                                           embedding_matrix=embedding_matrix,
                                                                           max_seq_len=max_seq_len,
                                                                           name='hypothesis_encoder').get_model()
        else:
            raise ValueError('Encoder not implemented')

        self.encoder_type = config['encoder']['type']
        self.path = config['path']
        self.classification_hidden_units = config['classification']['hidden_units']
        self.classification_dropout = config['classification']['dropout']
        self.n_classes = 3

        self.optimizer_type = config['training']['optimizer']
        self.init_lr = config['training']['init_lr']
        self.clipnorm = config['training']['clipnorm']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.model = None
        self.encoder_model = None
        self.optimizer = None
        self.get_model()

    def get_model(self):
        merged = Concatenate()([self.premise_encoder, self.hypothesis_encoder])
        classification_hidden = Dense(units=self.classification_hidden_units, activation='linear')(merged)
        classification_hidden = Dropout(self.classification_dropout)(classification_hidden)
        final = Dense(units=self.n_classes, activation='softmax')(classification_hidden)

        self.model = Model(inputs=[self.premise_input, self.hypothesis_input], output=final)
        print('NLI Classifier model -')
        print()
        print(self.model.summary())
        if self.optimizer_type == 'rmsprop':
            self.optimizer = optimizers.RMSprop(lr=self.init_lr, clipnorm=self.clipnorm)
        elif self.optimizer_type == 'adam':
            self.optimizer = optimizers.Adam(lr=self.init_lr, clipnorm=self.clipnorm)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optimizers.SGD(lr=self.init_lr, clipnorm=self.clipnorm)
        else:
            raise ValueError('Optimizer not implemented')

        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.encoder_model = K.function([self.model.get_layer('premise_encoder_input').input],
                                        [self.model.get_layer('premise_encoder').output])

    def fit(self, s1_train_x, s2_train_x, train_y,
            s1_dev_x, s2_dev_x, dev_y):
        lr_plateau = ReduceLROnPlateau(monitor='val_acc', factor=0.2, mode='max',
                                       patience=1, verbose=1, min_lr=0.0001)

        csv_logger = CSVLogger('{0}/{1}_training.log'.format(self.path,
                                                             self.encoder_type))

        checkpointer = ModelCheckpoint(filepath='{0}/{1}.hdf5'.format(self.path,
                                                                      self.encoder_type),
                                       monitor='val_acc', verbose=1, save_best_only=True)

        history = self.model.fit(x=[s1_train_x, s2_train_x],
                                 y=train_y,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=([s1_dev_x, s2_dev_x], dev_y),
                                 callbacks=[lr_plateau, csv_logger, checkpointer])
        return history

    def predict(self, s1_x, s2_x):
        return self.model.predict(x=[s1_x, s2_x])

    def encode(self, s1_x):
        return self.encoder_model([s1_x])[0]
