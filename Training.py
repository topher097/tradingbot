import pandas as pd
import numpy as np
from loggerSettings import logger
from BinanceConnect import *
import tensorflow as tf
from bot import *
from Technical import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

logger.debug('Tensorflow version: {}'.format(tf.__version__))

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

# https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2
# https://github.com/JanSchm/CapMarket/blob/master/bot_experiments/IBM_PriceFeatures.ipynb

class TrainingMisc():
    def __init__(self):
        #self.BiDirectionalLSTM  = BiDirectionalLSTM()
        self.methodFileName     = bot.methodFileName
        self.methods            = {}

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

    def loadMethodsJSON(self):
        # Load the methods and their values from JSON file
        try:
            with open(self.methodFileName, "r") as file:
                data            = file.read()
                self.methods    = json.loads(data)
            logger.info(f"Successfully loaded methods from {self.methodFileName}")
        except Exception as e:
            logger.error(f"Failed to load methods, error: {e}")

    def preprocessData(self, klines, pair, methods, plot=True):
        try:
            # Get the useful data from klines
            updatedKlines = klines[:,0:6]
            columnTitles = ["openTime", "open", "high", "low", "close", "volume"]
            # Create pandas data frame given kline data
            df = pd.DataFrame(updatedKlines, columns=columnTitles)

            """ add indicator(s) from methods to data frame """
            # Get RSI
            if 'RSI' in methods:   
                df['RSI']   = Technical.getRSI(self, klines, type=methods['RSI']['type'], timePeriod=methods['RSI']['timePeriod']).tolist()
            # Get Parabolic SAR
            if 'PSAR' in methods:   
                df['PSAR']  = Technical.getParabolicSAR(self, klines, acceleration=methods['PSAR']['acceleration'], maximum=methods['PSAR']['maximum']).tolist()
                df['PSAR']  = df['PSAR']/df['open']     # Get if PSAR is above or below open (above is >1, below is <1)      
            """ done with adding indicators """

            # Normalize and clean kline data
            df['volume'].replace(to_replace=0, method='ffill', inplace=True)        # Fix volume data
            df.sort_values('openTime', inplace=True)                                # Sort by ascending openTime
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
            logger.info('Training data shape: {}'.format(train_data.shape))
            logger.info('Validation data shape: {}'.format(val_data.shape))
            logger.info('Test data shape: {}'.format(test_data.shape))
            
            if plot: TrainingMisc.plotDailyChangesOfClosePricesAndVolume(self, pair, df_train, train_data, df_val, val_data, df_test, test_data)
            return df_train, train_data, df_val, val_data, df_test, test_data
        except Exception as e:
            logger.error(e)

    def plotDailyChangesOfClosePricesAndVolume(self, pair, df_train, train_data, df_val, val_data, df_test, test_data):
        fig = plt.figure(figsize=(15,10))
        st  = fig.suptitle(f"Data Separation for {pair}", fontsize=20)
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
    def __init__(self, train_data, val_data, test_data, modelFilePath, pair, plotEval):
        self.X_train, self.y_train  = [], []
        self.X_val, self.y_val      = [], []
        self.X_test, self.y_test    = [], []
        self.train_data = test_data
        self.val_data   = val_data
        self.test_data  = train_data
        self.modelFilePath = modelFilePath
        self.plotEval   = plotEval
        self.pair       = pair
        self.seq_len    = 128

        # gpus = tf.config.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         # Currently, memory growth needs to be the same across GPUs
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        #             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #             #print(tf.config.experimental.get_memory_info(gpu))
        #     except RuntimeError as e:
        #         # Memory growth must be set before GPUs have been initialized
        #         print(e)
        BiDirectionalLSTM.initChunks(self)
        BiDirectionalLSTM.runBiDirectionalLSTM(self)

    def runBiDirectionalLSTM(self):
        # Get or create and run the model
        if not os.path.exists(self.modelFilePath):
            logger.debug(f"New model file requested, running model training for: {self.modelFilePath}")
            try:
                logger.debug("Creating model")
                model = BiDirectionalLSTM.create_model(self)
                logger.debug(model.summary())
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
                callback = tf.keras.callbacks.ModelCheckpoint(self.modelFilePath, monitor='val_loss', save_best_only=True, verbose=1)
                model.fit(self.X_train, self.y_train,
                        batch_size=1024,
                        verbose=1,
                        callbacks=[callback],
                        epochs=200,
                        #shuffle=True,
                        validation_data=(self.X_val, self.y_val))  
                model = tf.keras.models.load_model(self.modelFilePath)
            except Exception as e:
                logger.error(f"Error while creating model, error: {e}")
        else:
            logger.debug(f"Model file already exists, loading file: {self.modelFilePath}")
            model = tf.keras.models.load_model(self.modelFilePath)

        logger.debug("Evaluating the model performance")
        print(model)
        
        """ Calc predictions and metrics """
        # Calculate the prediction for training, validation and test data
        train_predict   = model.predict(self.X_train)
        val_predict     = model.predict(self.X_val)
        test_predict    = model.predict(self.X_test)
        logger.debug("Prediction using data and model finished")

        # # Log evaluation metrics for all of the datasets
        # train_evaluate  = model.evaluate(self.X_train, self.y_train, batch_size=128, verbose=0)
        # val_evaluate    = model.evaluate(self.X_val, self.y_val, batch_size=128, verbose=0)
        # test_evaluate   = model.evaluate(self.X_test, self.X_test, batch_size=128, verbose=0)
        # logger.info('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_evaluate[0], train_evaluate[1], train_evaluate[2]))
        # logger.info('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_evaluate[0], val_evaluate[1], val_evaluate[2]))
        # logger.info('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_evaluate[0], test_evaluate[1], test_evaluate[2]))

        """ Plot the results """
        if self.plotEval:
            fig = plt.figure(figsize=(15,15))
            st  = fig.suptitle(f"Bi-Directional LSTM Model for {self.pair}", fontsize=18)
            st.set_y(1.02)
            
            # Plot the training data and results
            ax11 = fig.add_subplot(311)
            ax11.plot(self.train_data[:, 3], label="Closing returns")
            ax11.plot(train_predict, color="yellow", linewidth=3, label="Predicted Closing Returns")
            ax11.set_title("Training Data", fontsize=16)
            ax11.set_xlabel("Open Time")
            ax11.set_ylabel("Closing Returns")

            # Plot the validation data and results
            ax21 = fig.add_subplot(312)
            ax21.plot(self.val_data[:, 3], label="Closing returns")
            ax21.plot(val_predict, color="yellow", linewidth=3, label="Predicted Closing Returns")
            ax21.set_title("Validation Data", fontsize=16)
            ax21.set_xlabel("Open Time")
            ax21.set_ylabel("Closing Returns")

            # Plot the test data and results
            ax21 = fig.add_subplot(313)
            ax21.plot(self.test_data[:, 3], label="Closing returns")
            ax21.plot(test_predict, color="yellow", linewidth=3, label="Predicted Closing Returns")
            ax21.set_title("Test Data", fontsize=16)
            ax21.set_xlabel("Open Time")
            ax21.set_ylabel("Closing Returns")

            plt.tight_layout()
            plt.legend(loc='best')
            plt.draw()


    def initChunks(self):
        # Training data
        X_train, y_train = [], []
        for i in range(self.seq_len, len(self.train_data)):
            X_train.append(self.train_data[i-self.seq_len:i])      # Chunks of training data with a length of seq_len df-rows
            y_train.append(self.train_data[:, 3][i])               # Value of the 4th column (close price) of df-row seq_len+1
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

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
        logger.debug("Initialized chunks")

    def create_model(self):
        in_seq = Input(shape = (self.seq_len, self.test_data.shape[1]))
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
        logger.debug("Created model")
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
