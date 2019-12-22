import pickle
import numpy as np
import datetime as datetime

import torch

from ofdm.train_nn import train_nn
from ofdm.ofdm_functions import load_tx, gen_qpsk_data

#--- VARIABLES ---#

num_epochs = 1000
batch_size = np.power(2, 14)
learning_rate = .001
wmse_epsilon = 1e-4
expansion = 8 #Hidden layer expansion

qbits = 0
clipdb = 0

snrdb_low = 0
snrdb_high = 30

#--- LOAD AND GENERATE DATA ---#

timestamp = '20191221-154704'

# This returns encoded bits for a given number of samples.
# The number of samples is based on the OFDM size, bits per symbol (modulation)
# and the code rate
enc_bits, num_samples, ofdm_size, bits_per_symbol = load_tx(timestamp)

# Assumes bits_per_symbol = 2 aka QPSK
tx_signal, rx_signal, rx_symbols, rx_llrs, snrdb_list = gen_qpsk_data(enc_bits, snrdb_low, snrdb_high, ofdm_size)

#--- PREP DATA ---#

# Concatentate real and imaginary and reshape the data
input_samples = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)
input_samples = input_samples.reshape(-1, 2*ofdm_size)

# Add SNR to the input sample space
input_samples = np.concatenate((input_samples, np.power(10, snrdb_list/10)), axis=1)

# Reshape proper LLRs for output
output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
    
#--- TRAIN NETWORK ---#

llr_state_dict, optim_state_dict, train_loss = train_nn(input_samples, output_samples, ofdm_size, bits_per_symbol, expansion, num_epochs, batch_size, learning_rate, wmse_epsilon)

ts = datetime.datetime.now()

filename = ts.strftime('%Y%m%d-%H%M%S') + '_tx=' + timestamp + '_qbits={}_clipdb={}_snrdb={}-{}_epochs={}_lr={}_epsilon={}.pth'.format(qbits, clipdb, snrdb_low, snrdb_high, num_epochs, learning_rate, wmse_epsilon)        
    
filepath = 'outputs/model/' + filename
    
torch.save({
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'wmse_epsilon': wmse_epsilon,
        'expansion': expansion,
        
        'model_state_dict': llr_state_dict,
        'optimizer_state_dict': optim_state_dict,
        'train_loss': train_loss,
        
        'snrdb_low': snrdb_low,
        'snrdb_high': snrdb_high,
        'ofdm_size': ofdm_size,
        'tx_timestamp': timestamp,

        }, filepath)