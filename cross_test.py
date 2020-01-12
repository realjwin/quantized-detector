import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bp.parity import H, G
from ofdm.llr_nn import LLRestimator
from ofdm.ofdm_functions import encode_bits, gen_qpsk_data, gen_qpsk_qdata, decode_bits

#--- VARIABLES ---#

num_samples = np.power(2,14)

bp_iterations = 5
batch_size = 2**10
num_batches = num_samples // batch_size
clamp_value = 10

qbits = np.array([1, 3, 5])
clipdb = np.array([-10, -5, 0])

#for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--- LOAD TRAINED NETWORK ---#

trained_filename = ''
trained_filepath = 'outputs/model/' + trained_filename + '.pth'

checkpoint = torch.load(trained_filepath, map_location=device)

ofdm_size = checkpoint['ofdm_size']
expansion = checkpoint['expansion']
bits_per_symbol = 2 #hardcoding because forgot to include it

snrdb_low = checkpoint['snrdb_low']
snrdb_high = checkpoint['snrdb_high']

snrdb = np.linspace(snrdb_low, snrdb_high, 11)

#--- NN MODEL ---#

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs.")

LLRest = nn.DataParallel(LLRestimator(ofdm_size, bits_per_symbol, expansion))

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

uncoded_ber = np.zeros((len(qbits), len(clipdb), len(snrdb))
coded_ber = np.zeros(snrdb.shape)
coded_bler = np.zeros(snrdb.shape)

uncoded_ber_quantized = np.zeros((len(qbits), len(clipdb), len(snrdb))
coded_ber_quantized = np.zeros((len(qbits), len(clipdb), len(snrdb))
coded_bler_quantized = np.zeros((len(qbits), len(clipdb), len(snrdb))

uncoded_ber_nn = np.zeros((len(qbits), len(clipdb), len(snrdb))
coded_ber_nn = np.zeros((len(qbits), len(clipdb), len(snrdb))
coded_bler_nn = np.zeros((len(qbits), len(clipdb), len(snrdb))

for qbits_idx, qbits_val in enumerate(qbits):
    for clipdb_idx, clipdb_val in enumerate(clipdb):
            print('Q-Bits: {}, Clip: {} dB'.format(qbits_val, clipdb_val))
            
        for snrdb_idx, snrdb_val in enumerate(snrdb):
            print(snrdb_val)
    
        #--- GENERATE DATA ---#
        
        channel = None
        
        # Assumes bits_per_symbol = 2 aka QPSK
        tx_signal, rx_signal, rx_symbols, rx_llrs, snrdb_list = gen_qpsk_data(enc_bits, snrdb_val, snrdb_val, ofdm_size, channel)
        
        # Generate quantized data from the same SNR values already computed
        qrx_signal_rescaled, qrx_symbols, qrx_llrs = gen_qpsk_qdata(rx_signal, snrdb_list, qbits_val, clipdb_val, ofdm_size, channel)
        
        # Concatentate real and imaginary and reshape the data
        input_samples = np.concatenate((qrx_signal_rescaled.real.T, qrx_signal_rescaled.imag.T), axis=1)
        input_samples = input_samples.reshape(-1, 2*ofdm_size)
        
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
    
    wrong_qidx = np.where(np.sign(qrx_llrs) != np.sign(rx_llrs))
    wrong_idx = np.where(np.sign(llr_est) != np.sign(rx_llrs))
    #llr_est[wrong_idx] = rx_llrs[wrong_idx]
    #z_wrong = np.asarray([llr_est[wrong_idx], rx_llrs[wrong_idx]]).T
    
    wmse_quantized[snrdb_idx] = np.mean((qrx_llrs[wrong_qidx] - rx_llrs[wrong_qidx])**2 / (np.abs(rx_llrs[wrong_qidx]) + 10e-4))
    
    wmse_nn[snrdb_idx] = np.mean((llr_est[wrong_idx] - rx_llrs[wrong_idx])**2 / (np.abs(rx_llrs[wrong_idx]) + 10e-4))
    
    #compute number flipped? maybe later...
    
    #--- DECODING PERFORMANCE ---#
    
    cbits = (np.sign(rx_llrs) + 1) // 2
    bits = decode_bits(rx_llrs, H, bp_iterations, batch_size, clamp_value)
    
    cbits_nn = (np.sign(llr_est) + 1) // 2
    bits_nn = decode_bits(llr_est, H, bp_iterations, batch_size, clamp_value)
    
    cbits_quantized = (np.sign(qrx_llrs) + 1) // 2
    bits_quantized = decode_bits(qrx_llrs, H, bp_iterations, batch_size, clamp_value)
    
    uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
    coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
    
    uncoded_ber_nn[snrdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
    coded_ber_nn[snrdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_nn[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
    
    uncoded_ber_quantized[snrdb_idx] = np.mean(np.abs(cbits_quantized - enc_bits))
    coded_ber_quantized[snrdb_idx] = np.mean(np.abs(bits_quantized[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_quantized[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_quantized - enc_bits), axis=1)))

#--- SAVE CODED INFORMATION ---#
    
ber_path = 'outputs/ber/' + trained_filename + '.pkl'

with open(ber_path, 'wb') as f:
    save_dict = {
            'snrdb': snrdb,
            
            'uncoded_ber': uncoded_ber,
            'coded_ber': coded_ber,
            'coded_bler': coded_bler,
            
            'uncoded_ber_nn': uncoded_ber_nn,
            'coded_ber_nn': coded_ber_nn,
            'coded_bler_nn': coded_bler_nn,
            
            'uncoded_ber_quantized': uncoded_ber_quantized,
            'coded_ber_quantized': coded_ber_quantized,
            'coded_bler_quantized': coded_bler_quantized,
            
            'wmse_nn': wmse_nn,
            'wmse_quantized': wmse_quantized
            }
    
    pickle.dump(save_dict, f)


plot = True 
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle('NN Performance on Unquantized Inputs', fontsize=16, y=1.02)
             
    axes[0].semilogy(snrdb, uncoded_ber, label='Uncoded Traditional')
    axes[0].semilogy(snrdb, coded_ber, label='Coded Traditional')
    axes[0].semilogy(snrdb, uncoded_ber_nn, '--+', label='Uncoded NN')
    axes[0].semilogy(snrdb, coded_ber_nn, '--+', label='Coded NN')
    axes[0].semilogy(snrdb, uncoded_ber_quantized, '--*', label='Uncoded Quantized')
    axes[0].semilogy(snrdb, coded_ber_quantized, '--*', label='Coded Quantized')
    axes[0].set_title('BER')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('BER')
    axes[0].legend()
    
    axes[1].semilogy(snrdb, coded_bler, label='Traditional')
    axes[1].semilogy(snrdb, coded_bler_nn, '--+', label='NN')
    axes[1].semilogy(snrdb, coded_bler_quantized, '--*', label='Quantized')
    axes[1].set_title('BLER')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('BLER')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/plots/' + trained_filename + '_ber.eps', format='eps', bbox_inches='tight')

    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle('NN Performance on Unquantized Inputs', fontsize=16, y=1.02)
             
    axes[0].plot(snrdb, wmse_nn, '--+', label='NN WMSE')
    axes[0].plot(snrdb, wmse_quantized, '--*', label='Quantized WMSE')
    axes[0].set_title('WMSE')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('WMSE')
    axes[0].legend()
    
    axes[1].plot(train_loss, label='NN')
    axes[1].set_title('Train Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/plots/' + trained_filename + '_wmse.eps', format='eps', bbox_inches='tight')