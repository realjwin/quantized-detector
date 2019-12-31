import torch
import pickle
import numpy as np

import torch.nn as nn

from bp.bp import BeliefPropagation

def load_tx(timestamp):
    tx_file = timestamp + '_tx.pkl'
    tx_filepath = 'outputs/tx/' + tx_file
    
    with open(tx_filepath, 'rb') as f:
        data = pickle.load(f)    
        enc_bits = data['enc_bits']
        ofdm_size = data['ofdm_size']
        bits_per_symbol = data['bits_per_symbol']
        num_samples = data['num_samples']

    return enc_bits, num_samples, ofdm_size, bits_per_symbol

# Channel taps must be in frequency and be a matrix with each column
# corresponding to a channel for that OFDM symbol, so size: ofdm_size x num_samples
def gen_qpsk_data(bits, snrdb_low, snrdb_high, ofdm_size, channel=None):
    
    # Reshape bits into two columns for QPSK modulation
    bits = -2 * bits.reshape((-1, 2)) + 1 #(0 -> 1, 1 -> -1)
    
    # Compute QPSK modulation
    symbols = (1/np.sqrt(2))*bits[:,0] + (1j/np.sqrt(2))*bits[:,1]
    
    # Reshape back into a row
    tx_symbols = symbols.reshape((1, -1))
    
    # Compute number of OFDM symbols (samples)
    # There is always the assumption that tx_symbols
    # is a multiple of the OFDM size
    num_samples = np.size(tx_symbols) // ofdm_size
    
    # Generate a random SNR for each OFDM symbol (sample)
    snrdb = np.random.uniform(snrdb_low, snrdb_high, (num_samples, 1))

    # Make each column N (ofdm_size) symbols
    symbols = tx_symbols.reshape((-1, ofdm_size)).T
    
    # Multiply channel by the frequency taps if specified
    if channel is not None:
        symbols = channel * symbols
    
    # Create OFDM symbols
    ofdm_symbols = np.matmul(DFT(ofdm_size).conj().T, symbols)

    # Compute the SNR for each OFDM symbol
    snr_val = np.power(10, np.broadcast_to(snrdb.T, ofdm_symbols.shape)/10)

    # Generate noise for each element in ofdm_symbols
    noise = (np.random.normal(0, 1/np.sqrt(snr_val)) + 
             1j*np.random.normal(0, 1/np.sqrt(snr_val))) / np.sqrt(2)

    # Add the noise
    received_symbols = ofdm_symbols + noise

    # De-OFDM signal at the receiver
    deofdm_symbols = np.matmul(DFT(ofdm_size), received_symbols)
    
    if channel is not None:
        deofdm_symbols = deofdm_symbols / channel

    # Compute noise power for LLR computation
    # Note that noise power is 1/2 because this
    # is noise power per dimension
    if channel is not None:
        noise_power = .5 * (1 / (snr_val * np.abs(np.broadcast_to(channel, snr_val.shape))**2))
    else:
        noise_power = .5 * (1 / snr_val)

    # Compute log-likelihood ratios for QPSK, this happens per dimension
    # LLR is log(Pr=1 / Pr=0) (i.e. +inf = 1, -inf = -1)
    llr_bit0 = ( np.power(deofdm_symbols.real - 1/np.sqrt(2), 2) - 
                np.power(deofdm_symbols.real + 1/np.sqrt(2), 2) ) / (2*noise_power)
    llr_bit1 = ( np.power(deofdm_symbols.imag - 1/np.sqrt(2), 2) - 
                np.power(deofdm_symbols.imag + 1/np.sqrt(2), 2) ) / (2*noise_power)
      
    # Concatenate the LLRs together and then make them a 1-D list
    # because this is easiest to manipulate later
    llrs = np.concatenate((llr_bit0.T.reshape((-1,1)), llr_bit1.T.reshape((-1,1))), axis=1)
    llrs = llrs.reshape((1,-1))

    # Return values
    tx_signal = ofdm_symbols.T.reshape((1, -1))
    rx_signal = received_symbols.T.reshape((1, -1))
    rx_symbols = deofdm_symbols.T.reshape((1, -1))
    rx_llrs = llrs

    return tx_signal, rx_signal, rx_symbols, rx_llrs, snrdb

# Channel taps must be in frequency and be a vector of length ofdm_size.
# This channel must be the same as for the gen_qpsk_data
def gen_qpsk_qdata(rx_signal, snrdb_list, qbits, clipdb, ofdm_size, channel=None):
    rx_signal = rx_signal.reshape(-1, ofdm_size)
    
    # Convert clip ratio
    clip_ratio = np.power(10, (clipdb/10))       

    # Convert SNR
    snr = np.power(10, snrdb_list/10)
    
    # Set clip value (this is arbitrary), it is set
    # to 10 to avoid running off the float
    agc_clip = 10

    # Compute the average amplitude of the received signal
    # per dimension (real/imag). This assumes the signal
    # power is 1 and the noise power is 1/SNR. This computes
    # a sigma_rx for each OFDM symbol (sample) as they all
    # have different SNR values
    sigma_rx =  .5 * (1 + 1/snr)

    # Compute the agc scaling factor for each OFDM symbol (sample)
    factor = agc_clip / sigma_rx * clip_ratio
    
    # Scale signal by the scale factor, which is the same for every
    # element in the OFDM symbol
    rx_signal_scaled = np.broadcast_to(factor, rx_signal.shape) * rx_signal
    
    # Quantized the signal
    qrx_signal = quantizer(rx_signal_scaled, qbits, agc_clip)

    # Rescale the signal by the same factor, this is required
    # for accurate LLR values
    qrx_signal_rescaled = qrx_signal / np.broadcast_to(factor, qrx_signal.shape)
    
    # De-OFDM quantized signal at the receiver
    deofdm_qsymbols = np.matmul(DFT(ofdm_size), qrx_signal_rescaled.T)

    if channel is not None:
        deofdm_qsymbols = deofdm_qsymbols / channel

    # Compute the SNR for each OFDM symbol
    snr_val = np.power(10, np.broadcast_to(snrdb_list.T, deofdm_qsymbols.shape)/10)

    # Compute noise power for LLR computation
    # Note that noise power is 1/2 because this
    # is noise power per dimension
    if channel is not None:
        noise_power = .5 * (1 / (snr_val * np.abs(channel)**2))
    else:
        noise_power = .5 * (1 / snr_val)
    
    # Compute log-likelihood ratios for QPSK, this happens per dimension
    # LLR is log(Pr=1 / Pr=0) (i.e. +inf = 1, -inf = -1)
    qllr_bit0 = ( np.power(deofdm_qsymbols.real - 1/np.sqrt(2), 2) - 
                np.power(deofdm_qsymbols.real + 1/np.sqrt(2), 2) ) / (2*noise_power)
    qllr_bit1 = ( np.power(deofdm_qsymbols.imag - 1/np.sqrt(2), 2) - 
                np.power(deofdm_qsymbols.imag + 1/np.sqrt(2), 2) ) / (2*noise_power)
      
    qllrs = np.concatenate((qllr_bit0.T.reshape((-1,1)), qllr_bit1.T.reshape((-1,1))), axis=1)
    qllrs = qllrs.reshape((1,-1))
    
    
    # Return values, where each row is an OFDM symbol (sample)
    qrx_signal_rescaled = qrx_signal_rescaled.reshape((1,-1))
    qrx_symbols = deofdm_qsymbols.T.reshape((1,-1))
    qrx_llrs = qllrs
    
    return qrx_signal_rescaled, qrx_symbols, qrx_llrs

def DFT(N):  
    W = np.zeros((N, N), dtype=np.complex)
    
    for x in range(0,N):
        for y in range(0,N): 
            W[x,y] = np.exp(-1j*2*np.pi*x*y / N) / np.sqrt(N)
            
    return W

def create_bits(num_bits):
    return 
    
def encode_bits(num_bits, generator_matrix):
    # Generate bits
    bits = np.random.randint(2, size=num_bits).reshape((1,-1))
    
    # Reshape bits for encoding
    bits = bits.reshape((-1, generator_matrix.shape[1])).T
    
    # Encode bits
    cbits = np.mod(np.matmul(generator_matrix,bits),2)
    
    # Reshape back to list
    return cbits.T.reshape((1,-1))

def quantizer(inputs, num_bits, clip_value):
    num_levels = np.power(2, num_bits)
    step = 2*clip_value / (num_levels - 1)
    
    idx_real = np.floor(inputs.real/step + .5)
    idx_imag = np.floor(inputs.imag/step + .5)
    
    quantized_real = np.clip(step * idx_real, -(num_levels/2)*step+1, (num_levels/2)*step-1)
    quantized_imag = np.clip(step * idx_imag, -(num_levels/2)*step+1, (num_levels/2)*step-1)
    
    quantized_temp = np.zeros(inputs.shape, dtype=np.complex)
    quantized_temp.real = quantized_real
    quantized_temp.imag = quantized_imag

    return quantized_temp

def decode_bits(llrs, H, bp_iterations, batch_size, clamp_value):
    
    output_bits = np.zeros(llrs.shape)
    
    num_batches = llrs.shape[0] // batch_size
    
    #--- NN SETUP ---#
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bp_model = nn.DataParallel(BeliefPropagation(H, bp_iterations))
    else:
        bp_model = BeliefPropagation(H, bp_iterations)
    
    bp_model.eval()
    
    #send model to GPU
    bp_model.to(device)
        
    for batch in range(0, num_batches):
            start_idx = batch*batch_size
            end_idx =  (batch+1)*batch_size
                    
            llr = torch.tensor(llrs[start_idx:end_idx, :], dtype=torch.float, device=device)                            
            x = torch.zeros(llr.shape[0], np.sum(np.sum(H)), dtype=torch.float, device=device)
        
            y_est = bp_model(x, llr, clamp_value)
        
            output_bits[start_idx:end_idx, :] = np.round(y_est.cpu().detach().numpy())
            
    return output_bits