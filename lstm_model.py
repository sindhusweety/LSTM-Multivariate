import numpy as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from preprocessing import Preprocessing

class Model(object):
    def __init__(self):
        self.__obj = Preprocessing()
        self.__dataset = self.__obj._read_sim_utils()
        print(self.__dataset.head())
        self.premodeling()
    def premodeling(self):
        df = self.__dataset.replace('?', np.nan)
        print(df.isnull().sum())
        #df = df.astype('float32')
        # ADD DOWNSAMPLING FUNCTION
        #split dataset into train and test data in a 75% and 25% ratio of the instances
        print(len(df))
        first_half = round(len(df) * 0.75)
        second_half = round(len(df) * 0.25 )
        train_df, test_df = df[0 : first_half ], df[ second_half: -1]
        #Scaling the values
        train = train_df
        scalers = {}

        for i in train_df.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
            s_s = np.reshape(s_s, len(s_s))
            scalers['scaler_' + i] = scaler
            train[i] = s_s

        test = test_df
        for i in train_df.columns:
            scaler = scalers['scaler_' + i]
            s_s = scaler.transform(test[i].values.reshape(-1, 1))
            s_s = np.reshape(s_s, len(s_s))
            scalers['scaler_' + i] = scaler
            test[i] = s_s

        #Converting the series to samples

        def split_series(series, n_past, n_future):
            #
            # n_past ==> no of past observations
            #
            # n_future ==> no of future observations
            #
            X, y = list(), list()
            for window_start in range(len(series)):
                past_end = window_start + n_past
                future_end = past_end + n_future
                if future_end > len(series):
                    break
                # slicing the past and future parts of the window
                past, future = series[window_start:past_end, :], series[past_end:future_end, :]
                X.append(past)
                y.append(future)
            return np.array(X), np.array(y)

        # For this case, let's assume that
        # Given past 10 days observation, forecast the next 5 days observations.
        n_past = 10
        n_future = 5
        n_features = 4

        X_train, y_train = split_series(train.values, n_past, n_future)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))

        X_test, y_test = split_series(test.values, n_past, n_future)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

        #E1D1 ==> Sequence to Sequence Model with one encoder layer and one decoder layer.

        # E1D1
        # n_features ==> no of features at each timestep in the data.
        #
        encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
        encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
        encoder_outputs1 = encoder_l1(encoder_inputs)
        encoder_states1 = encoder_outputs1[1:]
        #
        decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
        #
        decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
        decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)
        #
        model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)
        #
        model_e1d1.summary()

        # E2D2
        # n_features ==> no of features at each timestep in the data.
        #
        encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
        encoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
        encoder_outputs1 = encoder_l1(encoder_inputs)
        encoder_states1 = encoder_outputs1[1:]
        encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
        encoder_outputs2 = encoder_l2(encoder_outputs1[0])
        encoder_states2 = encoder_outputs2[1:]
        #
        decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
        #
        decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
        decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
        decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
        #
        model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
        #
        model_e2d2.summary()


        #Training the models
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)

        model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
        history_e1d1 = model_e1d1.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32,
                                      verbose=0, callbacks=[reduce_lr])

        model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
        history_e2d2 = model_e2d2.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32,
                                      verbose=0, callbacks=[reduce_lr])

        plt.plot(history_e1d1.history['loss'])
        plt.plot(history_e1d1.history['val_loss'])
        plt.title("E1D1 Model Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Valid'])
        plt.savefig('E1D1.jpeg')

        plt.plot(history_e2d2.history['loss'])
        plt.plot(history_e2d2.history['val_loss'])
        plt.title("E2D2 Model Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Valid'])
        plt.savefig('E2D2.jpeg')

        #Prediction on test samples
        pred1_e1d1 = model_e1d1.predict(X_test)
        pred1_e2d2 = model_e2d2.predict(X_test)

        pred_e1d1 = model_e1d1.predict(X_train)
        pred_e2d2 = model_e2d2.predict(X_train)

        #Inverse Scaling of the predicted values
        for index, i in enumerate(train_df.columns):
            scaler = scalers['scaler_' + i]
            pred1_e1d1[:, :, index] = scaler.inverse_transform(pred1_e1d1[:, :, index])
            pred_e1d1[:, :, index] = scaler.inverse_transform(pred_e1d1[:, :, index])

            pred1_e2d2[:, :, index] = scaler.inverse_transform(pred1_e2d2[:, :, index])
            pred_e2d2[:, :, index] = scaler.inverse_transform(pred_e2d2[:, :, index])

            y_train[:, :, index] = scaler.inverse_transform(y_train[:, :, index])
            y_test[:, :, index] = scaler.inverse_transform(y_test[:, :, index])
        '''
        Checking Error
        Now we will calculate the mean absolute error of all observations.
        '''
        from sklearn.metrics import mean_absolute_error

        for index, i in enumerate(train_df.columns):
            print(i)
            for j in range(1, 6):
                print("Day ", j, ":")
                print("MAE-E1D1 : ", mean_absolute_error(y_test[:, j - 1, index], pred1_e1d1[:, j - 1, index]),
                      end=", ")
                print("MAE-E2D2 : ", mean_absolute_error(y_test[:, j - 1, index], pred1_e2d2[:, j - 1, index]))
            print()
            print()



Model()