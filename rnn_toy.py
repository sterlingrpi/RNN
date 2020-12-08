# this is a minimal example demonstrating an RNN that accepts sequences of
# variable lenght during training and during inference. Samples need to be
# of the same length within a single batch. But each batch can have samples
# with different lengths. Data must be 3D (sample num, time, data1, data2)

from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def get_model(num_input, num_output):
    num_units = 32
    # data must be 3D. Index 0 is sample number. 1 is time dimension. 2 & 3 are data
    input = Input(shape=(1, num_input, num_input))
    x = TimeDistributed(Flatten())(input)
    x = Dense(num_units)(x)
    x = LSTM(units=num_units, return_sequences=True)(x)
    x = LSTM(units=num_units, return_sequences=True)(x)
    x = LSTM(units=num_units)(x)
    output = Dense(num_output)(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mse', optimizer=Adam(lr=0.01))
    return model

def gen_data(num_cycles, num_samples):
    # number of points used in a cycle of the sin or saw wave
    resolution = 25
    sig_len = int(num_cycles*resolution)
    #initialize data arrays
    sigs = np.zeros((num_samples, sig_len, 1, 1))
    next = np.zeros((num_samples, 1))
    for i in range(num_samples):
        start = np.random.random()*2*np.pi
        time = np.linspace(start, start + num_cycles*2*np.pi, sig_len + 1)
        if np.random.random() > 0.5:
            sig = np.sin(time)
        else:
            sig = signal.sawtooth(time)
        sigs[i, :, 0, 0] = sig[:-1]
        next[i] = sig[-1]
    return sigs, next

if __name__ == '__main__':
    model = get_model(num_input=1, num_output=1)
    model.summary()

    sigs, next = gen_data(num_cycles=0.5, num_samples=10000)

    model.fit(x=sigs,
              y=next,
              epochs=3
              )

    for i in range(len(sigs)):
        prediction = model.predict(sigs[None, i])
        print('predicted next value =', prediction[0])
        print('actual next value =', next[i])
        plt.plot(sigs[i, :, 0, 0])
        plt.show()
