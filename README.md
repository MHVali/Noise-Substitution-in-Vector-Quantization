# **NSVQ: Noise Substitution in Vector Quantization for Machine Learning**

This repository contains PyTorch implementation of the NSVQ technique, which solves the gradient collapse problem when using Vector Quantization (VQ) function in a neural network. In other words, the technique passes gradients through VQ module when backpropagating through a neural network. The technique is published as the paper [NSVQ: Noise Substitution in Vector Quantization for Machine Learning](https://ieeexplore.ieee.org/abstract/document/9696322) in IEEE Access journal, January 2022. You can find **a short explanation of the paper** in [this medium post](https://medium.com/p/915f5814b5ce).

# **Contents of this repository**

- `NSVQ.py`: contains the main class of Noise Substitution in Vector Quantization (NSVQ)
- `train.py`: an example showing how to use and optimize NSVQ to learn a Normal distribution
- `plot_training_logs.py`: plots the training logs (which was saved druring execution of "train.py") in a pdf file

Due to some limitations of TensorBoard, we prefered our own custom logging function (plot_training_logs.py).

# **Required packages**
- Python (version: 3.8 or higher)
- PyTorch (version: 1.8.1)
- Numpy (version: 1.20.2)  
- Matplotlib (version: 3.6)

You can create the Python environment by passing the following lines of codes in your terminal window in the following order:

`conda create --name nsvq python=3.8`  
`conda activate nsvq`  
`pip install torch==1.8.1`  
`pip install numpy==1.20.2`  
`pip install matplotlib==3.6`

The requirements to use this repository is not that much strict, becuase the functions used in the code are so basic such that they also work with higher Python, PyTorch and Numpy versions.

# **Important: Codebook Replacement**

During training, we apply codebook replacement function (explained in section III.C in [the paper](https://ieeexplore.ieee.org/abstract/document/9696322)) to discard those codebook vectors which are not involved in the vector quantization process. There are two reasons for that; 1) the codebook replacement acts as a trigger to make the codebook vectors to start updating in some applications , 2) it allows exploiting from all available codebook vectors (better vector quantization perfromance) and avoiding isolated and rarely used codebooks. The codebook replacement is implemented as a function named `replace_unused_codebooks` in the NSVQ.py class. The essential explanations are prepared for this function in the code. Feel free to change the **discarding_threshod** and **num_batches** parameters which are related to the codebook replacement function based on your application. However, the recommended procedure of codebook replacement is in the following.

Call this function after a specific number of training batches (**num_batches**) during training. In the beginning, the number of replaced codebooks might increase (the number of replaced codebooks will be printed out during training). However, the main trend must be decreasing after some training time, because the codebooks will find a location inside the distribution which makes them useful representative of the distribution. If the replacement trend is not decreasing, increase the **num_batches** or decrease the **discarding_threshold**. Stop calling the function at the latest stages of training (for example the last 1000 training batches) in order not to introduce any new codebook entry which would not have the right time to be tuned and optimized until the end of training. Remember that the number of repalced codebook vectors will be printed after each round you call the function.

# **Results directory**

The "Results" directory contains the values of objective metrics, which were used to plot the figures (Fig1 and Fig3) of the paper. The values are provided in JSON file format, and refer to PESQ, pSNR, and STOI metrics for the speech coding and SSIM and PeakSNR metrics for image compression scenario. We have shared these results with the aim of saving time for reproducabiltiy and making it easier for researchers to do potential comparisons. Note that in order to calculate the PESQ values, we have installed and used the **PESQ package from PyPI** ([under this link](https://pypi.org/project/pesq/)) in our Python environment.

# **Abstract of the paper**

Machine learning algorithms have been shown to be highly effective in solving optimization problems in a wide range of applications. Such algorithms typically use gradient descent with backpropagation and the chain rule. Hence, the backpropagation fails if intermediate gradients are zero for some functions in the computational graph, because it causes the gradients to collapse when multiplying with zero. Vector quantization is one of those challenging functions for machine learning algorithms, since it is a piece-wise constant function and its gradient is zero almost everywhere. A typical solution is to apply the straight through estimator which simply copies the gradients over the vector quantization function in the backpropagation. Other solutions are based on smooth or stochastic approximation. This study proposes a vector quantization technique called NSVQ, which approximates the vector quantization behavior by substituting a multiplicative noise so that it can be used for machine learning problems. Specifically, the vector quantization error is replaced by product of the original error and a normalized noise vector, the samples of which are drawn from a zero-mean, unit-variance normal distribution. We test our proposed NSVQ in three scenarios with various types of applications. Based on the experiments, the proposed NSVQ achieves more accuracy and faster convergence in comparison to the straight through estimator, exponential moving averages, and the MiniBatchKmeans approaches.

# **Cite the paper as**

Mohammad Hassan Vali and Tom Bäckström, “NSVQ: Noise Substitution in Vector Quantization for Machine Learning,” IEEE Access, vol. 10, pp. 13598–13610, 2022.

```bibtex
@article{vali2022nsvq,
  title={NSVQ: Noise Substitution in Vector Quantization for Machine Learning},
  author={Vali, Mohammad Hassan and Bäckström, Tom},
  journal={IEEE Access},
  volume={10},
  pages={13598--13610},
  year={2022},
  publisher={IEEE}
}
```
