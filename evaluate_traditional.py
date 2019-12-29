import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bp.parity import H, G
from ofdm.ofdm_functions import encode_bits, gen_qpsk_data, gen_qpsk_qdata, decode_bits

#--- VARIABLES ---#

num_samples = np.power(2,18)

bp_iterations = 5
batch_size = 2**10
num_batches = num_samples // batch_size
clamp_value = 10

ofdm_size = 32
bits_per_symbol = 2
snrdb_low = 0
snrdb_high = 30
snrdb = np.linspace(snrdb_low, snrdb_high,11)
qbits = np.array([1,3,5])
clipdb = np.array([-6, -3, 0, 3, 6])

#--- GENERATE BITS ---#

code_rate = .5 #hardcoding
num_bits = num_samples * code_rate * bits_per_symbol * ofdm_size

if num_bits.is_integer():
    num_bits = np.int(num_bits) 

    enc_bits = encode_bits(num_bits, G)

#--- COMPUTE PERFORMANCE ---#

for qbits_idx, qbits_val in enumerate(qbits):
    for clipdb_idx, clipdb_val in enumerate(clipdb):
        print('Q-Bits: {}, Clip: {} dB'.format(qbits_val, clipdb_val))
            
        uncoded_ber = np.zeros(snrdb.shape)
        coded_ber = np.zeros(snrdb.shape)
        coded_bler = np.zeros(snrdb.shape)
        
        uncoded_ber_quantized = np.zeros(snrdb.shape)
        coded_ber_quantized = np.zeros(snrdb.shape)
        coded_bler_quantized = np.zeros(snrdb.shape)
        
        for snrdb_idx, snrdb_val in enumerate(snrdb):
            print(snrdb_val)
            
            #--- GENERATE DATA ---#
            
            channel = None #(np.random.normal(0, 1, (ofdm_size,1)) + np.random.normal(0, 1, (ofdm_size,1))) / np.sqrt(2)
            
            # Assumes bits_per_symbol = 2 aka QPSK
            tx_signal, rx_signal, rx_symbols, rx_llrs, snrdb_list = gen_qpsk_data(enc_bits, snrdb_val, snrdb_val, ofdm_size, channel)
            
            # Generate quantized data from the same SNR values already computed
            qrx_signal_rescaled, qrx_symbols, qrx_llrs = gen_qpsk_qdata(rx_signal, snrdb_list, qbits_val, clipdb_val, ofdm_size, channel)
        
            # Reshape values for training 
            qrx_llrs = qrx_llrs.reshape(-1, 2*ofdm_size)
            rx_llrs = rx_llrs.reshape(-1, 2*ofdm_size)
            enc_bits = enc_bits.reshape(-1, 2*ofdm_size)
        
            #--- DECODING PERFORMANCE ---#
            
            cbits = (np.sign(rx_llrs) + 1) // 2
            bits = decode_bits(rx_llrs, H, bp_iterations, batch_size, clamp_value)
         
            cbits_quantized = (np.sign(qrx_llrs) + 1) // 2
            bits_quantized = decode_bits(qrx_llrs, H, bp_iterations, batch_size, clamp_value)
            
            uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
            coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
            coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
            
            uncoded_ber_quantized[snrdb_idx] = np.mean(np.abs(cbits_quantized - enc_bits))
            coded_ber_quantized[snrdb_idx] = np.mean(np.abs(bits_quantized[:, 0:32] - enc_bits[:, 0:32]))
            coded_bler_quantized[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_quantized - enc_bits), axis=1)))
        
        #--- SAVE CODED INFORMATION ---#
            
        ber_path = 'outputs/ber/quantized_qbits={}_clipdb={}_snrdb={}-{}.pkl'.format(qbits_val, clipdb_val, snrdb_low, snrdb_high)
        
        with open(ber_path, 'wb') as f:
            save_dict = {
                    'snrdb': snrdb,
                    
                    'uncoded_ber': uncoded_ber,
                    'coded_ber': coded_ber,
                    'coded_bler': coded_bler,
                    
                    'uncoded_ber_quantized': uncoded_ber_quantized,
                    'coded_ber_quantized': coded_ber_quantized,
                    'coded_bler_quantized': coded_bler_quantized,
                    }
            
            pickle.dump(save_dict, f)