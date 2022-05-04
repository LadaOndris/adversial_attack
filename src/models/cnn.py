from tensorflow.keras import Model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D
from tensorflow.python.keras.models import load_model

from src.models.base import ModelProvider


class CnnModelProvider(ModelProvider):

    def get_model(self) -> Model:
        model = Sequential([
            InputLayer([28, 28, 1]),
            Conv2D(filters=48, kernel_size=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Dropout(0.2),
            Conv2D(filters=96, kernel_size=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        return model


class TrainedModelProvider(ModelProvider):

    def get_model(self) -> Model:
        return load_model('weights/classifier')
