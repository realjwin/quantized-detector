import pickle
import numpy as np
import datetime as datetime

import torch

from ofdm.train_nn_channel import train_nn_channel
from ofdm.ofdm_functions import load_tx, gen_qpsk_data, gen_qpsk_qdata

#--- VARIABLES ---#

num_epochs = 2500
batch_size = np.power(2, 14)
learning_rate = .001
wmse_epsilon = ([.1, 1, 5])

qbits = np.array([1, 3, 5])
clipdb = np.array([-10, -5, 0])

#--- LOAD PRETRAINED NETWORK ---#

pretrained_filename = '20191231-120423_tx=20191221-213020_qbits=0_clipdb=0_snrdb=0-30_epochs=1000_lr=0.001_epsilon=0.1_channel'
pretrained_filepath = 'outputs/model/' + pretrained_filename + '.pth'


checkpoint = torch.load(pretrained_filepath)

ofdm_size = checkpoint['ofdm_size']
expansion = checkpoint['expansion']

tx_timestamp = checkpoint['tx_timestamp']
snrdb_low = checkpoint['snrdb_low']
snrdb_high = checkpoint['snrdb_high']


#--- LOAD AND GENERATE DATA ---#

enc_bits, num_samples, ofdm_size, bits_per_symbol = load_tx(tx_timestamp)

# Create channels
channel = (np.random.normal(0, 1, (ofdm_size, num_samples)) + np.random.normal(0, 1, (ofdm_size,num_samples))) / np.sqrt(2)

# Assumes bits_per_symbol = 2 aka QPSK
tx_signal, rx_signal, rx_symbols, rx_llrs, snrdb_list = gen_qpsk_data(enc_bits, snrdb_low, snrdb_high, ofdm_size, channel)

#--- TRAIN QUANTIZED ---#

for wmse_epsilon_idx, wmse_epsilon_val in enumerate(wmse_epsilon):
    for qbits_idx, qbits_val in enumerate(qbits):
        for clipdb_idx, clipdb_val in enumerate(clipdb):
            print('Q-Bits: {}, Clip: {} dB, WMSE Epsilon: {}'.format(qbits_val, clipdb_val, wmse_epsilon_val))
            
            #--- GENERATE AND PREP QUANTIZED DATA ---#
            
            # Generate quantized data from the same SNR values already computed
            qrx_signal_rescaled, qrx_symbols, qrx_llrs = gen_qpsk_qdata(rx_signal, snrdb_list, qbits_val, clipdb_val, ofdm_size, channel)
            
            # Concatentate real and imaginary and reshape the data
            input_samples = np.concatenate((qrx_signal_rescaled.real.T, qrx_signal_rescaled.imag.T), axis=1)
            input_samples = input_samples.reshape(-1, 2*ofdm_size)
            
            # Add chanel to innput sample space
            channel_samples = channel.T.reshape(1,-1)
            channel_samples = np.concatenate((channel_samples.real.T, channel_samples.imag.T), axis=1)
            channel_samples = channel_samples.reshape(-1, 2*ofdm_size)
            input_samples = np.concatenate((input_samples, channel_samples), axis=1)
            
            # Add SNR to the input sample space
            input_samples = np.concatenate((input_samples, np.power(10, snrdb_list/10)), axis=1)
            
            # Reshape proper LLRs for output
            output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
                
            #--- TRAIN NETWORK ---#
            
            llr_state_dict, optim_state_dict, train_loss = train_nn_channel(input_samples, output_samples, ofdm_size, bits_per_symbol, expansion, num_epochs, batch_size, learning_rate, wmse_epsilon_val, pretrained_filename)
            
            ts = datetime.datetime.now()
            
            filename = ts.strftime('%Y%m%d-%H%M%S') + '_tx=' + tx_timestamp + '_qbits={}_clipdb={}_snrdb={}-{}_epochs={}_lr={}_epsilon={}_channel.pth'.format(qbits_val, clipdb_val, snrdb_low, snrdb_high, num_epochs, learning_rate, wmse_epsilon_val)        
                
            filepath = 'outputs/model/' + filename
                
            torch.save({
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'wmse_epsilon': wmse_epsilon_val,
                    'expansion': expansion,
                    
                    'pretrained_filename': pretrained_filename,
                    
                    'qbits': qbits_val,
                    'clipdb': clipdb_val,
                    
                    'model_state_dict': llr_state_dict,
                    'optimizer_state_dict': optim_state_dict,
                    'train_loss': train_loss,
                    
                    'snrdb_low': snrdb_low,
                    'snrdb_high': snrdb_high,
                    'ofdm_size': ofdm_size,
                    'num_samples': num_samples,
                    'bits_per_symbol': bits_per_symbol,
                    'tx_timestamp': tx_timestamp,
            
                    }, filepath)