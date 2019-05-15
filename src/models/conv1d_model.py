from keras import regularizers
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv1D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


class Conv1dModel:

    @staticmethod
    def build_model(input_shape=(25, 3),
                    cnn_blocks=3,
                    num_filters=3,
                    kernel_size=5,
                    dropout=0.1,
                    learning_rate=1e-3,
                    weight_decay=0,
                    activation='relu'):
        model = Sequential()
        model.add(Conv1D(num_filters,
                         kernel_size=kernel_size,
                         kernel_regularizer=regularizers.l2(weight_decay),
                         padding='same',
                         input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        #        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout))

        for i in range(cnn_blocks - 1):
            model.add(Conv1D(num_filters * (i+1),
                             kernel_size=kernel_size,
                             kernel_regularizer=regularizers.l2(weight_decay)
                             # padding='same'
                             ))
            model.add(BatchNormalization())
            model.add(Activation(activation))
            #            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout))
        # FC
        model.add(Flatten())
        # output
        model.add(Dense(3, activation='softmax'))
        model.compile(
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
            optimizer=Adam(lr=learning_rate))
        return model
