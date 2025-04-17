import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class PricePredictor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_advanced_model()
        
    def _build_advanced_model(self):
        # پارامترهای پیشرفته برای مدل LSTM
        n_features = 10  # تعداد ویژگی‌های ورودی
        sequence_length = self.config.get("sequence_length", 60)
        
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True, 
                                activation="tanh"),
                                input_shape=(sequence_length, n_features)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        
        return model
