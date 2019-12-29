import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

#--- LOAD RESULTS ---#

script_path = os.path.dirname(os.path.abspath(__file__))
ber_path = os.path.join(script_path, 'outputs', 'ber')

ber_nn_dict = dict()
ber_quantized_dict = dict()
# r=root, d=directories, f = files
for r, d, f in os.walk(ber_path):
    for file in f:
        if '.pth' in file:
            file = file[:-8]
            qbits = np.float(file.split('_')[2].split('=')[1])
            clipdb = np.float(file.split('_')[3].split('=')[1])
            snrdb_low = np.float(file.split('_')[4].split('=')[1].split('-')[0])
            snrdb_high = np.float(file.split('_')[4].split('=')[1].split('-')[1])
            wmse_epsilon = np.float(file.split('_')[7].split('=')[1])
            
            if (qbits, clipdb) in ber_nn_dict:
                ber_nn_dict[(qbits, clipdb)].append((file + '.pth.pkl', wmse_epsilon))
            else: 
                ber_nn_dict[(qbits, clipdb)] = [(file + '.pth.pkl', wmse_epsilon)]
        elif file.split('_')[0] == 'quantized':
            qbits = np.float(file.split('_')[1].split('=')[1])
            clipdb = np.float(file.split('_')[2].split('=')[1])

            ber_quantized_dict[(qbits, clipdb)] = file

#--- PLOT ---#

# Plot for each Q-Bits and Clip dB levels
for key, ber_results in ber_nn_dict.items():
    
    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Q-Bits: {}, Clip: {} dB'.format(key[0], key[1]), fontsize=16, y=1.02)
    
    # Load traditional curves
    traditional_filename = ber_quantized_dict[key]
    
    ber_path = 'outputs/ber/' + traditional_filename

    with open(ber_path, 'rb') as f:
        data = pickle.load(f)
        
        snrdb = data['snrdb']
        
        uncoded_ber = data['uncoded_ber']
        coded_ber = data['coded_ber']
        coded_bler = data['coded_bler']

        uncoded_ber_quantized = data['uncoded_ber_quantized']
        coded_ber_quantized = data['coded_ber_quantized']
        coded_bler_quantized = data['coded_bler_quantized']
    
    # Plot traditional
    axes[0].semilogy(snrdb, uncoded_ber, label='Uncoded Traditional')
    axes[0].semilogy(snrdb, coded_ber, label='Coded Traditional')
    axes[0].semilogy(snrdb, uncoded_ber_quantized, '--*', label='Uncoded Quantized Traditional')
    axes[0].semilogy(snrdb, coded_ber_quantized, '--*', label='Coded Quantized Traditional')
    axes[1].semilogy(snrdb, coded_bler_quantized, '--*', label='Quantized Traditional')
    axes[1].semilogy(snrdb, coded_bler, label='Traditional')
    
    # Load neural network curves
    for ber_result in ber_results:
        filename = ber_result[0]
        wmse_epsilon = ber_result[1]
        
        ber_path = 'outputs/ber/' + filename
    
        with open(ber_path, 'rb') as f:
            data = pickle.load(f)
            
            snrdb = data['snrdb']
            uncoded_ber_nn = data['uncoded_ber_nn']
            coded_ber_nn = data['coded_ber_nn']
            coded_bler_nn = data['coded_bler_nn']
            train_loss = data['train_loss']
            
        axes[0].semilogy(snrdb, uncoded_ber_nn, '--+', label='Uncoded NN, \u03B5 = {}'.format(wmse_epsilon))
        axes[0].semilogy(snrdb, coded_ber_nn, '--+', label='Coded NN, \u03B5 = {}'.format(wmse_epsilon))
        axes[1].semilogy(snrdb, coded_bler_nn, '--+', label='NN, \u03B5 = {}'.format(wmse_epsilon))
        axes[2].plot(train_loss, label='NN, \u03B5 = {}'.format(wmse_epsilon))
    
    # Set axes
    axes[0].set_title('BER')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('BER')
    axes[0].legend()
    
    axes[1].set_title('BLER')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('BLER')
    axes[1].legend()
    
    axes[2].set_title('Train Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('WMSE Loss')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/plots/qbits={}_clipdb={}.eps'.format(key[0], key[1]), format='eps', bbox_inches='tight')