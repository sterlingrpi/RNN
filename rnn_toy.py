# this is a minimal example demonstrating an RNN that accepts sequences of
# variable lenght during training and during inference. Samples need to be
# of the same length within a single batch. But each batch can have samples
# with different lengths. Data must be 3D (sample_num, time, data1, data2)

from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def get_model(num_input, num_output):
    num_units = 32
    # data must be 3D. Index 0 is sample number. 1 is time dimension. 2 & 3 are data
    input = Input(shape=(None, num_input, 1))
    x = TimeDistributed(Flatten())(input)
    x = Dense(num_units)(x)
    x = LSTM(units=num_units, return_sequences=True)(x)
    x = LSTM(units=num_units, return_sequences=True)(x)
    x = LSTM(units=num_units)(x)
    output = Dense(num_output)(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mse', optimizer=Adam(lr=0.01))
    return model

# resolution is number of points used in a cycle of the sin or saw wave
def gen_data(num_samples, num_cycles, resolution, num_next):
    sig_len = int(num_cycles*resolution)
    #initialize data arrays
    sigs = np.zeros((num_samples, sig_len, 1, 1))
    next = np.zeros((num_samples, num_next))
    for i in range(num_samples):
        start = np.random.random()*2*np.pi
        time = np.linspace(start, start + num_cycles*2*np.pi, sig_len + num_next)
        random_num = np.random.random()
        if random_num > 0.75:
            sig = np.sin(time)
        elif random_num > 0.50:
            sig = signal.sawtooth(time)
        elif random_num > 0.25:
            sig = -1*signal.sawtooth(time)
        else:
            sig = signal.square(time)
        sigs[i, :, 0, 0] = sig[:-num_next]
        next[i] = sig[-num_next:]
    return sigs, next

if __name__ == '__main__':
    num_predict = 18
    model = get_model(num_input=1, num_output=num_predict)
    model.summary()

    sigs, next = gen_data(num_samples=10000, num_cycles=2, resolution=18, num_next=num_predict)

    model.fit(x=sigs,
              y=next,
              epochs=1,
              shuffle=True,
              )

    sigs, next = gen_data(num_samples=10000, num_cycles=1, resolution=18, num_next=num_predict)

    model.fit(x=sigs,
              y=next,
              epochs=1,
              shuffle=True,
              )

    sigs, _ = gen_data(num_samples=100, num_cycles=2, resolution=18, num_next=num_predict)

    for i in range(len(sigs)):
        prediction = model.predict(sigs[None, i])
        plt.plot(sigs[i, :, 0, 0])
        plt.plot(np.arange(prediction.shape[1]) + sigs.shape[1], prediction[0])
        plt.show()
