import pandas as pd
import numpy as np
from loggerSettings import logger
from BinanceConnect import *
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
logger.debug('Tensorflow version: {}'.format(tf.__version__))

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

# https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2

class Training():
    def __init__(self):
        self.BiDirectionalLSTM = BiDirectionalLSTM()

    def calcPercentChange(self, klines):
        try:
            df = pd.DataFrame(klines, columns = ["openTime", "open", "high", "low", "close", "volume", "closeTime", "assetVolume", "numTrades", "takerBuyBaseAssetVolume", "takerBuyQuoteAssetVolume", "ignore"])
            openPercentChange   = df['open'].pct_change()
            highPercentChange   = df['high'].pct_change()
            lowPercentChange    = df['low'].pct_change()
            closePercentChange  = df['close'].pct_change()
            volPercentChange    = df['volume'].pct_change()
            return openPercentChange, highPercentChange, lowPercentChange, closePercentChange, volPercentChange
        except Exception as e:
            logger.error(e)

    def preprocessData(self, klines):
        try:
            # Create pandas data frame given kline data
            df = pd.DataFrame(klines[:,0:6], columns = ["openTime", "open", "high", "low", "close", "volume"])
            df['volume'].replace(to_replace=0, method='ffill', inplace=True)
            df.sort_values('openTime', inplace=True)
            # Percent change
            df['open']      = df['open'].pct_change()
            df['high']      = df['high'].pct_change()
            df['low']       = df['low'].pct_change()
            df['close']     = df['close'].pct_change()
            df['volume']    = df['volume'].pct_change()
            df.dropna(how='any', axis=0, inplace=True)      # Drop all rows with NaN values
            # Min-max normalization
            min_return      = min(df[['open', 'high', 'low', 'close']].min(axis=0))
            max_return      = max(df[['open', 'high', 'low', 'close']].max(axis=0))
            df['open']      = (df['open'] - min_return) / (max_return - min_return)
            df['high']      = (df['high'] - min_return) / (max_return - min_return)
            df['low']       = (df['low'] - min_return) / (max_return - min_return)
            df['close']     = (df['close'] - min_return) / (max_return - min_return)
            min_volume      = df['volume'].min(axis=0)
            max_volume      = df['volume'].max(axis=0)
            df['volume']    = (df['volume'] - min_volume) / (max_volume - min_volume)
            # Split into training periods
            times           = sorted(df.index.values)
            last_10pct      = sorted(df.index.values)[-int(0.1*len(times))]     # Last 10% of series
            last_20pct      = sorted(df.index.values)[-int(0.2*len(times))]     # Last 20% of series
            df_train        = df[(df.index < last_20pct)]                       # Training data are 80% of total data
            df_val          = df[(df.index >= last_20pct) & (df.index < last_10pct)]
            df_test         = df[(df.index >= last_10pct)]
            # Remove openTime column
            df_train.drop(columns=['openTime'], inplace=True)
            df_val.drop(columns=['openTime'], inplace=True)
            df_test.drop(columns=['openTime'], inplace=True)
            # Convert pandas columns into arrays
            train_data      = df_train.values
            val_data        = df_val.values
            test_data       = df_test.values
            print('Training data shape: {}'.format(train_data.shape))
            print('Validation data shape: {}'.format(val_data.shape))
            print('Test data shape: {}'.format(test_data.shape))
            Training.plotDailyChangesOfClosePricesAndVolume(self, df_train, train_data, df_val, val_data, df_test, test_data)
            return df_train, train_data, df_val, val_data, df_test, test_data
        except Exception as e:
            logger.error(e)

    def plotDailyChangesOfClosePricesAndVolume(self, df_train, train_data, df_val, val_data, df_test, test_data):
        fig = plt.figure(figsize=(15,10))
        st  = fig.suptitle("Data Separation", fontsize=20)
        st.set_y(0.92)

        ax1 = fig.add_subplot(211)
        ax1.plot(np.arange(train_data.shape[0]), df_train['close'], label='Training data')
        ax1.plot(np.arange(train_data.shape[0], train_data.shape[0]+val_data.shape[0]), df_val['close'], label='Validation data')
        ax1.plot(np.arange(train_data.shape[0]+val_data.shape[0], train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), df_test['close'], label='Test data')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Closing Returns')

        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(train_data.shape[0]), df_train['volume'], label='Training data')
        ax2.plot(np.arange(train_data.shape[0], train_data.shape[0]+val_data.shape[0]), df_val['volume'], label='Validation data')
        ax2.plot(np.arange(train_data.shape[0]+val_data.shape[0], train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), df_test['volume'], label='Test data')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Normalized Volume Changes')

        plt.legend(loc='best')
        plt.draw()


    def calcSharpeRatio(self):
        pass

    

class BiDirectionalLSTM():
    def __init__(self):
        self.X_train, self.y_train  = [], []
        self.X_val, self.y_val      = [], []
        self.X_test, self.y_test    = [], []
        self.train_data = Training.train_data
        self.val_data   = Training.val_data
        self.test_data  = Training.test_data
        self.seq_len    = Training.seq_len

    def runBiDirectionalLSTM(self):
        # Get the model
        model = BiDirectionalLSTM.create_model(self)
        logger.debug(model.summary())
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        callback = tf.keras.callbacks.ModelCheckpoint('Bi-LSTM.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
        model.fit(self.X_train, self.y_train,
                  batch_size=2048,
                  verbose=2,
                  callbacks=[callback],
                  epochs=200,
                  #shuffle=True,
                  validation_data=(self.X_val, self.y_val),)    
        model = tf.keras.models.load_model('/content/Bi-LSTM.hdf5')

        """ Calc predictions and metrics """
        # Calculate the predication for training, validation and test data




    def initChunks(self):
        # Training data
        X_train, y_train = [], []
        for i in range(self.seq_len, len(self.train_data)):
            X_train.append(self.train_data[i-self.seq_len:i])      # Chunks of training data with a length of seq_len df-rows
            y_train.append(self.train_data[:, 3][i])               # Value of the 4th column (close price) of df-row seq_len+1
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

        # Validation data
        X_val, y_val = [], []
        for i in range(self.seq_len, len(self.val_data)):
            X_val.append(self.val_data[i-self.seq_len:i])
            y_val.append(self.val_data[:, 3][i])
        self.X_val, self.y_val = np.array(X_val), np.array(y_val)

        # Test data
        X_test, y_test = [], []
        for i in range(self.seq_len, len(self.test_data)):
            X_test.append(self.test_data[i-self.seq_len:i])
            y_test.append(self.test_data[:, 3][i])
        self.X_test, self.y_test = np.array(X_test), np.array(y_test)

    def create_model(self):
        BiDirectionalLSTM.initChunks(self)
        in_seq = Input(shape = (self.seq_len, 5))
        x = Bidirectional(LSTM(128, return_sequences=True))(in_seq)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x) 

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        out = Dense(1, activation="linear")(conc)      

        model = Model(inputs=in_seq, outputs=out)
        model.compile(loss="mse", optimizer="adam", metrics=['mae', 'mape'])    
        return model


    def Inception_A(layer_in, c7):
        branch1x1_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch1x1 = BatchNormalization()(branch1x1_1)
        branch1x1 = ReLU()(branch1x1)
        
        branch5x5_1 = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(layer_in)
        branch5x5 = BatchNormalization()(branch5x5_1)
        branch5x5 = ReLU()(branch5x5)
        branch5x5 = Conv1D(c7, kernel_size=5, padding='same', use_bias=False)(branch5x5)
        branch5x5 = BatchNormalization()(branch5x5)
        branch5x5 = ReLU()(branch5x5)  
        
        branch3x3_1 = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(layer_in)
        branch3x3 = BatchNormalization()(branch3x3_1)
        branch3x3 = ReLU()(branch3x3)
        branch3x3 = Conv1D(c7, kernel_size=3, padding='same', use_bias=False)(branch3x3)
        branch3x3 = BatchNormalization()(branch3x3)
        branch3x3 = ReLU()(branch3x3)
        branch3x3 = Conv1D(c7, kernel_size=3, padding='same', use_bias=False)(branch3x3)
        branch3x3 = BatchNormalization()(branch3x3)
        branch3x3 = ReLU()(branch3x3) 
        
        branch_pool = AveragePooling1D(pool_size=(3), strides=1, padding='same')(layer_in)
        branch_pool = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(branch_pool)
        branch_pool = BatchNormalization()(branch_pool)
        branch_pool = ReLU()(branch_pool)
        outputs = Concatenate(axis=-1)([branch1x1, branch5x5, branch3x3, branch_pool])
        return outputs


    def Inception_B(layer_in, c7):
        branch3x3 = Conv1D(c7, kernel_size=3, padding="same", strides=2, use_bias=False)(layer_in)
        branch3x3 = BatchNormalization()(branch3x3)
        branch3x3 = ReLU()(branch3x3)  
        
        branch3x3dbl = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch3x3dbl = BatchNormalization()(branch3x3dbl)
        branch3x3dbl = ReLU()(branch3x3dbl)  
        branch3x3dbl = Conv1D(c7, kernel_size=3, padding="same", use_bias=False)(branch3x3dbl)  
        branch3x3dbl = BatchNormalization()(branch3x3dbl)
        branch3x3dbl = ReLU()(branch3x3dbl)  
        branch3x3dbl = Conv1D(c7, kernel_size=3, padding="same", strides=2, use_bias=False)(branch3x3dbl)    
        branch3x3dbl = BatchNormalization()(branch3x3dbl)
        branch3x3dbl = ReLU()(branch3x3dbl)   
        
        branch_pool = MaxPooling1D(pool_size=3, strides=2, padding="same")(layer_in)
        
        outputs = Concatenate(axis=-1)([branch3x3, branch3x3dbl, branch_pool])
        return outputs


    def Inception_C(layer_in, c7):
        branch1x1_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch1x1 = BatchNormalization()(branch1x1_1)
        branch1x1 = ReLU()(branch1x1)   
        
        branch7x7_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)
        branch7x7 = BatchNormalization()(branch7x7_1)
        branch7x7 = ReLU()(branch7x7)   
        branch7x7 = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7)
        branch7x7 = BatchNormalization()(branch7x7)
        branch7x7 = ReLU()(branch7x7)  
        branch7x7 = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7)  
        branch7x7 = BatchNormalization()(branch7x7)
        branch7x7 = ReLU()(branch7x7)   

        branch7x7dbl_1 = Conv1D(c7, kernel_size=1, padding="same", use_bias=False)(layer_in)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl_1)
        branch7x7dbl = ReLU()(branch7x7dbl)  
        branch7x7dbl = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl) 
        branch7x7dbl = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl)  
        branch7x7dbl = Conv1D(c7, kernel_size=(7), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl)  
        branch7x7dbl = Conv1D(c7, kernel_size=(1), padding="same", use_bias=False)(branch7x7dbl)  
        branch7x7dbl = BatchNormalization()(branch7x7dbl)
        branch7x7dbl = ReLU()(branch7x7dbl)  
        
        branch_pool = AveragePooling1D(pool_size=3, strides=1, padding='same')(layer_in)
        branch_pool = Conv1D(c7, kernel_size=1, padding='same', use_bias=False)(branch_pool)
        branch_pool = BatchNormalization()(branch_pool)
        branch_pool = ReLU()(branch_pool)  
        
        outputs = Concatenate(axis=-1)([branch1x1, branch7x7, branch7x7dbl, branch_pool])
        return outputs
