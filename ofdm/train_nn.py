import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ofdm.llr_nn import LLRestimator, weighted_mse

def train_nn(input_samples, output_samples, ofdm_size, bits_per_symbol, expansion, num_epochs, batch_size, learning_rate, wmse_epsilon, load_model=None):
    
    #--- VARIABLES ---#
    
    num_samples = input_samples.shape[0]
    num_batches = num_samples // batch_size

    #--- INIT NN ---#

    # Check for GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print if using GPUs
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs.")
        
    # Parallelize model for multiple GPUs
    LLRest = nn.DataParallel(LLRestimator(ofdm_size, bits_per_symbol, expansion))
    
    # Send model to GPUs
    LLRest.to(device)

    # Set optimizer
    #optimizer = optim.SGD(LLRest.parameters(), lr=learning_rate)
    optimizer = optim.Adam(LLRest.parameters(), lr=learning_rate, amsgrad=True)

    #--- LOAD MODEL ---#
    
    # Load pre-trained model if there is one
    if load_model:
        model_path = 'outputs/model/' + load_model + '.pth'
        
        checkpoint = torch.load(model_path, map_location=device)
        
        LLRest.load_state_dict(checkpoint['model_state_dict'])

    #--- TRAINING ---#
    
    # Store train loss for each epoch
    train_loss = np.zeros(num_epochs)

    for epoch in range(0, num_epochs):
        
        # Shuffle data in each epoch
        p = np.random.permutation(num_samples)
        input_samples = input_samples[p]
        output_samples = output_samples[p]
        
        for batch in range(0, num_batches):
            start_idx = batch*batch_size
            end_idx = (batch+1)*batch_size
            
            # Setup data in batch, each row is a sample
            x_batch = torch.tensor(input_samples[start_idx:end_idx], dtype=torch.float, requires_grad=True, device=device)
            y_batch = torch.tensor(output_samples[start_idx:end_idx], dtype=torch.float, device=device)
            y_batch = torch.tanh(y_batch)
            
            y_est= LLRest(x_batch)
            
            # Compute loss and gradients
            loss = weighted_mse(y_est, y_batch, wmse_epsilon)
            loss.backward()
            
            # Add to the epoch loss
            train_loss[epoch] += loss.item() / num_batches
            
            # Update weights with stored gradients
            # then clear stored gradients
            optimizer.step()
            optimizer.zero_grad()
            
            # Cleanup tensors to free space (probably unnecessary)
            del x_batch
            del y_batch
            del y_est
            del loss
        
        # Print statistics
        if np.mod(epoch, 10) == 0:
            # Don't store gradients on this pass
            with torch.no_grad():
                # Choose random sample from training set
                random_sample = np.random.choice(num_samples, np.power(2, 10))
                
                # Setup data for neural network
                x_test = torch.tensor(input_samples[random_sample], dtype=torch.float, device=device)
                y_test = torch.tensor(output_samples[random_sample], dtype=torch.float, device=device)
                y_test = torch.tanh(y_test)
                
                y_est = LLRest(x_test)
                test_loss = weighted_mse(y_est, y_test, wmse_epsilon)
            
            # Compute bits from outputs, by sending tensor
            # to CPU, detaching from the network gradient tracker
            # and converting to a numpy array
            y_est_bits = np.sign(y_est.cpu().detach().numpy())
            y_bits = np.sign(output_samples[random_sample])
            
            # Compute number of flipped values
            num_flipped = np.mean(np.abs(y_est_bits - y_bits) // 2)
            temp = output_samples[random_sample]
            flipped_values= np.abs(temp[np.where(np.abs(y_est_bits - y_bits))])
            
            # Print validation information
            if num_flipped == 0:
                print('no test values are flipped')
            else:
                print('flipped mean: {}, median: {}, max: {}'.format(np.mean(flipped_values), np.median(flipped_values), np.amax(flipped_values)))
    
            print('[epoch %d] train_loss: %.3f, test_loss: %.3f, flipped_ber: %.3f' % (epoch + 1, train_loss[epoch], test_loss, num_flipped))
            
            del x_test
            del y_test
            del y_est
            del test_loss
    
    # Return model parameters
    return LLRest.state_dict(), optimizer.state_dict(), train_loss