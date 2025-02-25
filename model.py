import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
import tensorflow as tf

def Trans_unet(input_size = (128,128,8,4)):
    inputs = Input(input_size)
    
    conv1 = Conv3D(64, 3, activation='relu', padding = 'same')(inputs)
    conv1 = Conv3D(64, 3, activation='relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv2 = Conv3D(128, 3, activation='relu', padding = 'same')(pool1)
    conv2 = Conv3D(128, 3, activation='relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    conv3 = Conv3D(256, 3, activation='relu', padding = 'same')(pool2)
    conv3 = Conv3D(256, 3, activation='relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    #transformer
    bottleneck = Conv3D(256, 3, activation='relu', padding='same')(pool3)
    bottleneck = Reshape((-1, 256))(bottleneck)

    #position
    seq_len = bottleneck.shape[1]
    pos_encoding = positional_encoding(seq_len, 256)
    bottleneck += pos_encoding

    for _ in range(12):
        bottleneck = transformer_block(bottleneck, 8, 256)

    bottleneck = Reshape((16, 16, 1, 256))(bottleneck)

    conv5 = Conv3D(512, 3, activation='relu', padding = 'same')(bottleneck)
    conv5 = Conv3D(512, 3, activation='relu', padding = 'same')(conv5)

    conv9 = Conv3DTranspose(256, 2, strides=2, padding='same')(conv5)
    up9 = concatenate([conv9, conv3], axis=-1)
    conv9 = Conv3D(256, 3, activation='relu', padding = 'same')(up9)
    conv9 = Conv3D(256, 3, activation='relu', padding = 'same')(conv9)

    conv10 = Conv3DTranspose(128, 2, strides=2, padding='same')(conv9)
    up10 = concatenate([conv10, conv2], axis=-1)
    conv10 = Conv3D(128, 3, activation='relu', padding = 'same')(up10)
    conv10 = Conv3D(128, 3, activation='relu', padding = 'same')(conv10)

    conv11 = Conv3DTranspose(64, 2, strides=2, padding='same')(conv10)
    up11 = concatenate([conv11, conv1], axis=-1)
    conv11 = Conv3D(64, 3, activation='relu', padding = 'same')(up11)
    conv11 = Conv3D(64, 3, activation='relu', padding = 'same')(conv11)

    conv11 = Conv3D(4, 1, padding = 'same')(conv11)
 
    model = Model(inputs, conv11)

    model.compile(optimizer = Adam(), loss = MeanAbsoluteError())
    model.summary()

    return model

#Transformer
def transformer_block(input, num_heads, embed_dim):

    x_norm = LayerNormalization(epsilon=1e-6)(input)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x_norm, x_norm)
    
    attn_output = Add()([input, attn_output])
    
    x_norm = LayerNormalization(epsilon=1e-6)(attn_output)
    x_ffn = Dense(embed_dim, activation="relu")(x_norm)
    x_ffn = Dense(input.shape[-1])(x_ffn)
    
    output = Add()([attn_output, x_ffn])
    
    return output

#Position
def positional_encoding(seq_len, d_model):
    pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = pos * angle_rates

    sines = tf.sin(angle_rads[:, 0::2])

    cosines = tf.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[tf.newaxis, ...]
    return pos_encoding





