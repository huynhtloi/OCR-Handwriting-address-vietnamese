# import our model, different layers and activation function
from keras.layers import Layer, Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation, RepeatVector, Permute, multiply, InputSpec
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical, plot_model
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau# import our model, different layers and activation function
import numpy as np

char_list = [' ','#','%','\'','(',')','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':','?','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','a','b','c','d','e','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','Â','Ê','Ô','à','á','â','ã','è','é','ê','ì','í','ò','ó','ô','õ','ù','ú','ý','Ă','ă','Đ','đ','ĩ','ũ','Ơ','ơ','ư','ạ','ả','Ấ','ấ','ầ','ẩ','ậ','ắ','ằ','ẵ','ặ','ẻ','ẽ','ế','ề','ể','ễ','ệ','ỉ','ị','ọ','ỏ','ố','ồ','ổ','ỗ','ộ','ớ','ờ','ở','ỡ','ợ','ụ','ủ','Ứ','ứ','ừ','ử','ữ','ự','ỳ','ỵ','ỷ','ỹ']

def model():
    # input with shape of height=32 and width=128
    inputs = Input(shape=(118,2167,1))

    # Block 1
    x = Conv2D(64, (3,3), padding='same')(inputs)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_1 = x

    # Block 2
    x = Conv2D(128, (3,3), padding='same')(x)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_2 = x

    # Block 3
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_3 = x

    # Block4
    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,x_3])
    x = Activation('relu')(x)
    x_4 = x

    # Block5
    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_5 = x

    # Block6
    x = Conv2D(512, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x,x_5])
    x = Activation('relu')(x)

    # Block7
    x = Conv2D(1024, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 1))(x)
    x = Activation('relu')(x)

    # pooling layer with kernel size (3,1) to make the height/3 #(1,9,512) to (1, 3, 512)
    x = MaxPool2D(pool_size=(3, 1))(x)

    # # to remove the first dimension of one: (1, 3, 512) to (3, 512)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)

    # # # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.25))(squeezed)
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.25))(blstm_1)

    # # this is our softmax character proprobility with timesteps
    outputs = Dense(145, activation = 'softmax')(blstm_2)

    # model to be used at test time

    return Model(inputs, outputs)

def predict_results(prediction):
    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                            greedy=True)[0][0])
    # see the results
    all_predictions = []
    i = 0
    for x in out:
        print("predicted text = ", end = '')
        pred = ""
        for p in x:
            if int(p) != -1:
                pred += char_list[int(p)]
        all_predictions.append(pred)
        i += 1
    return all_predictions