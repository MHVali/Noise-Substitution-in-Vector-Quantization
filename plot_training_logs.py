"""
Code to plot the training logs saved during execution of the code "train.py". The plots will be saved as a pdf file.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# hyper-parameters you used for training. Now they are needed to load your saved arrays.
batch_size = 64
vq_bitrate = 8
learning_rate = 1e-3

# create pdf file
pdf_file = PdfPages(f'log_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.pdf')

# loading the training logs
load_address = './output/'
total_vq_loss = np.load(load_address + f'total_vq_loss_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy')
total_perplexity = np.load(load_address + f'total_perplexity_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy')
with open(load_address + f"codebook_usage_list_{vq_bitrate}bits_bs{batch_size}_lr{learning_rate}", "rb") as fp:
    codebook_usage_list = pickle.load(fp)

num_of_logs = np.size(total_vq_loss)

# plotting used codebook indices during training
num_bars = int(2**vq_bitrate)
for i in range(len(codebook_usage_list)):
    histogram = np.log10(codebook_usage_list[i] + 1)
    fig = plt.figure(figsize=(15,5))
    plt.bar(np.arange(1, num_bars + 1), height=histogram, width=1)
    plt.title(f'Codebook Usage Histogram During Training | VQ Bitrate = {vq_bitrate}')
    plt.xlabel('Codebook Index')
    plt.ylabel('log10(codebook usage)')
    pdf_file.savefig(fig, bbox_inches='tight')

# plotting VQ loss during training
fig = plt.figure(figsize=(15, 5))
total_vq_loss = total_vq_loss.reshape(-1,1)
plt.plot(total_vq_loss)
plt.title(f'VQ Loss')
plt.xlabel('Training Iterations')
plt.ylabel('Mean Squared Error')
pdf_file.savefig(fig, bbox_inches='tight')

# plotting perplexity (average codebook usage per batch) during training
fig = plt.figure(figsize=(15, 5))
total_perplexity = total_perplexity.reshape(-1,1)
plt.plot(total_perplexity)
plt.title('Perplexity')
plt.xlabel('Training Iterations')
plt.ylabel('Average Codebook Usage')
pdf_file.savefig(fig, bbox_inches='tight')

pdf_file.close()
