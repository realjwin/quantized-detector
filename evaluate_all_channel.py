import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bp.parity import H, G
from ofdm.llr_nn import LLRestimator_channel
from ofdm.ofdm_functions import encode_bits, gen_qpsk_data, gen_qpsk_qdata, decode_bits

#--- VARIABLES ---#

num_samples = np.power(2,12)

bp_iterations = 5
batch_size = 2**10
num_batches = num_samples // batch_size
clamp_value = 10

#for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--- LOAD RESULTS ---#

script_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_path, 'outputs', 'model')

filenames = []
# r=root, d=directories, f = files
for r, d, f in os.walk(model_path):
    for file in f:
        if '.pth' in file:
            qbits = np.float(file.split('_')[2].split('=')[1])
            
            # If it's not the unquantized file
            if qbits != 0:
                filenames.append(file)

num_filenames = len(filenames)

for idx, trained_filename in enumerate(filenames):
    print('Testing file ({}/{}): {}'.format(idx, num_filenames, trained_filename))
    
    #--- LOAD TRAINED NETWORK ---#

    trained_filepath = 'outputs/model/' + trained_filename
    
    checkpoint = torch.load(trained_filepath, map_location=device)
    
    snrdb_low = checkpoint['snrdb_low']
    snrdb_high = checkpoint['snrdb_high']
    snrdb = np.linspace(snrdb_low, snrdb_high, 11)
    
    ofdm_size = checkpoint['ofdm_size']
    expansion = checkpoint['expansion']
    bits_per_symbol = 2 #hardcoding because forgot to include it, checkpoint['bits_per_symbol']
    
    qbits = checkpoint['qbits']
    clipdb = checkpoint['clipdb']
    wmse_epsilon = checkpoint['wmse_epsilon']
    
    train_loss = checkpoint['train_loss']
    
    #--- NN MODEL ---#
    
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs.")
    
    LLRest = nn.DataParallel(LLRestimator_channel(ofdm_size, bits_per_symbol, expansion))
    
    LLRest.eval()
    
    #send model to GPU
    LLRest.to(device)
    
    LLRest.load_state_dict(checkpoint['model_state_dict'])
    
    #--- GENERATE BITS ---#
    
    code_rate = .5 #hardcoding
    num_bits = num_samples * code_rate * bits_per_symbol * ofdm_size
    
    if num_bits.is_integer():
        num_bits = np.int(num_bits) 
    
        enc_bits = encode_bits(num_bits, G)
    
    #--- COMPUTE PERFORMANCE ---#
    
    uncoded_ber_nn = np.zeros(snrdb.shape)
    coded_ber_nn = np.zeros(snrdb.shape)
    coded_bler_nn = np.zeros(snrdb.shape)
    wmse_nn = np.zeros(snrdb.shape)
    
    for snrdb_idx, snrdb_val in enumerate(snrdb):
        print(snrdb_val)
        
        #--- GENERATE DATA ---#
        
        channel = (np.random.normal(0, 1, (ofdm_size, num_samples)) + np.random.normal(0, 1, (ofdm_size,num_samples))) / np.sqrt(2)
        
        # Assumes bits_per_symbol = 2 aka QPSK
        tx_signal, rx_signal, rx_symbols, rx_llrs, snrdb_list = gen_qpsk_data(enc_bits, snrdb_val, snrdb_val, ofdm_size, channel)
        
        # Generate quantized data from the same SNR values already computed
        qrx_signal_rescaled, qrx_symbols, qrx_llrs = gen_qpsk_qdata(rx_signal, snrdb_list, qbits, clipdb, ofdm_size, channel)
        
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
    
        # Reshape other values for training 
        qrx_llrs = qrx_llrs.reshape(-1, 2*ofdm_size)
        rx_llrs = rx_llrs.reshape(-1, 2*ofdm_size)
        enc_bits = enc_bits.reshape(-1, 2*ofdm_size)
    
        #--- INFERENCE ---#
        
        llr_est = np.zeros(output_samples.shape)
        
        for batch in range(0, num_batches):
            start_idx = batch*batch_size
            end_idx =  (batch+1)*batch_size
            
            x_input = torch.tensor(input_samples[start_idx:end_idx], dtype=torch.float, device=device)
            
            with torch.no_grad():
                llr_est_temp = LLRest(x_input)
                llr_est_temp = 0.5*torch.log((1+llr_est_temp)/(1-llr_est_temp))
                
            llr_est[start_idx:end_idx, :] = np.clip(llr_est_temp.cpu().detach().numpy(), -clamp_value, clamp_value)
        
        #--- LLR WMSE PERFORMANCE ---#
        
        wmse_nn[snrdb_idx] = np.mean((llr_est - rx_llrs)**2 / (np.abs(rx_llrs) + wmse_epsilon))
        
        #--- DECODING PERFORMANCE ---#
        
        cbits_nn = (np.sign(llr_est) + 1) // 2
        bits_nn = decode_bits(llr_est, H, bp_iterations, batch_size, clamp_value)
        
        uncoded_ber_nn[snrdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
        coded_ber_nn[snrdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
        coded_bler_nn[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
    
    #--- SAVE CODED INFORMATION ---#
        
    ber_path = 'outputs/ber/' + trained_filename + '.pkl'
    
    with open(ber_path, 'wb') as f:
        save_dict = {
                'snrdb': snrdb,
                
                'uncoded_ber_nn': uncoded_ber_nn,
                'coded_ber_nn': coded_ber_nn,
                'coded_bler_nn': coded_bler_nn,
                
                'wmse_nn': wmse_nn,
                
                'train_loss': train_loss
                }
        
        pickle.dump(save_dict, f)