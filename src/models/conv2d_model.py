from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam


class Conv2dModel:

    @staticmethod
    def build_model(input_shape=(25, 2, 3),
                    cnn_blocks=3,
                    num_filters=3,
                    kernel_size=(5, 1),
                    dropout=0.2,
                    learning_rate=1e-3,
                    weight_decay=0.01,
                    activation='relu'):
        model = Sequential()
        model.add(Conv2D(num_filters,
                         kernel_size=kernel_size,
                         kernel_regularizer=regularizers.l2(weight_decay),
                         padding='same',
                         input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(rate=1 - dropout))

        for i in range(cnn_blocks - 1):
            model.add(Conv2D(2 ** (i + 3),
                             kernel_size=kernel_size,
                             kernel_regularizer=regularizers.l2(weight_decay),
                             padding='same'
                             ))
            model.add(BatchNormalization())
            model.add(Activation(activation))
            model.add(MaxPooling2D(pool_size=(2, 1)))
            model.add(Dropout(rate = 1 - dropout))
        # FC
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
            optimizer=Adam(lr=learning_rate))
        return model
