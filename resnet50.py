from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Dropout, Add, Flatten, Input
from keras.initializers import glorot_uniform 
from tensorflow.keras.regularizers import l2

def identity_block(x, filters, l2_r= 0):
    
    x_shortcut = x
    f1, f2, f3 = filters
    
    #subblock 1
    x = Conv2D(kernel_size=(1,1), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), strides=(1,1), filters=(f1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #subblock 2
    x = Conv2D(kernel_size=(3,3), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), strides=(1,1), filters=(f2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #subblock 3
    x = Conv2D(kernel_size=(1,1), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), strides=(1,1), filters=(f3), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x

def convolutional_block(x, filters, s=2, l2_r= 0):
    
    x_shortcut = x
    f1, f2, f3 = filters
    
    #conv layer 1
    x = Conv2D(kernel_size=(1,1), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), strides=(s,s), filters=(f1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #conv layer 2
    x = Conv2D(kernel_size=(3,3), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), strides=(1,1), filters=(f2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #conv layer 3
    x = Conv2D(kernel_size=(1,1), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), strides=(1,1), filters=(f3), padding='valid')(x)
    x = BatchNormalization()(x)
    
    #conv layer 4
    x_shortcut = Conv2D(kernel_size=(1,1), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), strides=(s,s), filters=(f3), padding='valid')(x)
    x_shortcut = BatchNormalization()(x)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x

def MyResNet50(input_shape=(224, 224, 3), drop_rate=0.5, l2_r=0):
    
    x_input = Input(input_shape)

    x = Conv2D(filters=64, kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=l2(l2_r), kernel_size=(7,7), strides=(2,2), padding='same')(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = convolutional_block(x, filters=[64, 64, 256], s=1, l2_r=l2_r)
    x = Dropout(drop_rate)(x)
    x = identity_block(x, filters=[64, 64, 256], l2_r=l2_r)
    x = identity_block(x, filters=[64, 64, 256], l2_r=l2_r)
    x = identity_block(x, filters=[64, 64, 256], l2_r=l2_r)

    x = convolutional_block(x, filters=[128, 128, 512], s=2, l2_r=l2_r)
    x = Dropout(drop_rate)(x)
    x = identity_block(x, filters=[128, 128, 512], l2_r=l2_r)
    x = identity_block(x, filters=[128, 128, 512], l2_r=l2_r)
    x = identity_block(x, filters=[128, 128, 512], l2_r=l2_r) 

    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=glorot_uniform(seed=1))(x)
    x = Dropout(0.40)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=1))(x)

    model = Model(inputs=x_input, outputs=x)
    
    return model