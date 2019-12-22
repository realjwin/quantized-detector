import pickle
import numpy as np
import datetime as datetime

from bp.parity import G
from ofdm.ofdm_functions import encode_bits

#--- VARIABLES ---#

ts = datetime.datetime.now()

ofdm_size = 32
bits_per_symbol = 2
code_rate = .5

num_samples = np.power(2,20)

num_bits = num_samples * code_rate * bits_per_symbol * ofdm_size

if num_bits.is_integer():
    num_bits = np.int(num_bits) 

    #--- GENERATE BITS ---#
    
    enc_bits = encode_bits(num_bits, G)
    
    filename = 'outputs/tx/' + ts.strftime('%Y%m%d-%H%M%S') + '_tx.pkl'
    
    with open(filename, 'wb') as f:
        save_dict = {
                'enc_bits': enc_bits,
                'ofdm_size': ofdm_size,
                'bits_per_symbol': bits_per_symbol,
                'num_samples': num_samples
                }
    
        pickle.dump(save_dict, f)
        
    print(filename)
    
else:
    print('Number of bits is not an integer.')