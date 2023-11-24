"""
An example code to show how to train the NSVQ module on a Normal distribution.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from NSVQ import NSVQ
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
vq_bitrate = 8
embedding_dim = 128
learning_rate = 1e-3
batch_size = 64
num_training_batches = 100000 #number of training updates
normal_mean = 0 #mean for normal distribution
normal_std = 1 #standard deviation for normal distribution
training_log_batches = 500 #number of batches to get logs of training
replacement_num_batches = 1000 #number of batches to check codebook activity and discard inactive codebook vectors
num_of_logs = int(num_training_batches / training_log_batches)

# Arrays to save the logs of training
total_vq_loss = [] # tracks VQ loss
total_perplexity = [] # tracks perplexity
used_codebook_indices = np.zeros((int(2**vq_bitrate),)) # tracks indices of used codebook vectors

vector_quantizer = NSVQ(int(2**vq_bitrate), embedding_dim, device=device)
vector_quantizer.to(device)

optimizer = optim.Adam(vector_quantizer.parameters(), lr=learning_rate)

vq_loss_accumulator = perplexity_accumulator = 0

for iter in range(num_training_batches):

    data_batch = torch.normal(normal_mean, normal_std, size=(batch_size, embedding_dim)).to(device)

    optimizer.zero_grad()

    quantized_batch, perplexity, selected_indices = vector_quantizer(data_batch)
    vq_loss = F.mse_loss(data_batch, quantized_batch)

    vq_loss.backward()
    optimizer.step()

    unique_selected_indices = np.unique(selected_indices.cpu().detach().numpy())
    used_codebook_indices[unique_selected_indices] += 1

    # codebook replacement
    if ((iter + 1) % replacement_num_batches == 0) & (iter <= num_training_batches - 2*replacement_num_batches):
        vector_quantizer.replace_unused_codebooks(replacement_num_batches)

    vq_loss_accumulator += vq_loss.item()
    perplexity_accumulator += perplexity.item()

    # save and print logs
    if (iter+1) % training_log_batches == 0:
        vq_loss_average = vq_loss_accumulator / training_log_batches
        perplexity_average = perplexity_accumulator / training_log_batches
        total_vq_loss.append(vq_loss_average)
        total_perplexity.append(perplexity_average)
        vq_loss_accumulator = perplexity_accumulator = 0
        print("training iter:{}, vq loss:{:.6f}, perpexlity:{:.2f}".format(iter + 1, vq_loss_average, perplexity_average))


save_address = './output/'
os.makedirs(save_address)
np.save(save_address + f'total_vq_loss_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', np.asarray(total_vq_loss))
np.save(save_address + f'total_perplexity_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', np.asarray(total_perplexity))
np.save(save_address + f'codebook_usage_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', used_codebook_indices)

checkpoint_state = {"vector_quantizer": vector_quantizer.state_dict()}
torch.save(checkpoint_state, save_address + f"vector_quantizer_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.pt")

print("\nTraining Finished >>> Logs and Checkpoints Saved!!!")

######################## Evaluation (Inference) of NSVQ #############################

data = torch.normal(normal_mean, normal_std, size=(2**18, embedding_dim)).to(device)
quantized_data = torch.zeros_like(data)

eval_batch_size = 64
num_batches = int(data.shape[0]/eval_batch_size)
with torch.no_grad():
    for i in range(num_batches):
        data_batch = data[(i*eval_batch_size):((i+1)*eval_batch_size)]
        quantized_data[(i*eval_batch_size):((i+1)*eval_batch_size)] = vector_quantizer.inference(data_batch)

mse = F.mse_loss(data, quantized_data).item()
print("Mean Squared Error = {:.4f}".format(mse))







