from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import set_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)
train_generator = train_datagen.flow_from_directory(
    './train/',
    color_mode='rgb',
    target_size=(128, 128),
    batch_size=16)


input_tensor = Input(shape=(128, 128, 3))
x = input_tensor
for f in [64, 128]:
    x = Conv2D(f, 3, 2, 'valid', activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output_tensor = Dense(3, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
model.summary()

ee = EarlyStopping('loss', min_delta=0.005, patience=2, restore_best_weights=True)
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
model.fit_generator(train_generator, epochs=100, callbacks=[ee], shuffle=True)

model.save('model.h5', include_optimizer=False)
