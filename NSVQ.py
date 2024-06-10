import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist

class NSVQ(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=torch.device('cpu'), discarding_threshold=0.01, initialization='normal'):
        super(NSVQ, self).__init__()

        """
        Inputs:
        
        1. num_embeddings = Number of codebook entries
        
        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)
        
        3. device = The device which executes the code (CPU or GPU)
        
        ########## change the following inputs based on your application ##########
        
        4. discarding_threshold = Percentage threshold for discarding unused codebooks
        
        5. initialization = Initial distribution for codebooks

        """

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12

        if initialization == 'normal':
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)
        elif initialization == 'uniform':
            codebooks = uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings).sample([self.num_embeddings, self.embedding_dim])
        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")

        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

        # Counter variable which contains the number of times each codebook is used
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=device)

    def forward(self, input_data):

        """
        This function performs the main proposed vector quantization function using NSVQ trick to pass the gradients.
        Use this forward function for training phase.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for training | shape: (NxD) )
                perplexity (average usage of codebook entries)
        """

        # compute the distances between input and codebooks vectors
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, self.codebooks.t()))
                     + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True))

        min_indices = torch.argmin(distances, dim=1)

        hard_quantized_input = self.codebooks[min_indices]
        random_vector = normal_dist.Normal(0, 1).sample(input_data.shape).to(self.device)

        norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector

        quantized_input = input_data + vq_error

        # claculating the perplexity (average usage of codebook entries)
        encodings = torch.zeros(input_data.shape[0], self.num_embeddings, device=input_data.device)
        encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        with torch.no_grad():
            self.codebooks_used[min_indices] += 1

        # use the first returned tensor "quantized_input" for training phase (Notice that you do not have to use the
        # tensor "quantized_input" for inference (evaluation) phase)
        # Also notice you do not need to add a new loss term (for VQ) to your global loss function to optimize codebooks.
        # Just return the tensor of "quantized_input" as vector quantized version of the input data.
        return quantized_input, perplexity, self.codebooks_used.cpu().numpy()


    def replace_unused_codebooks(self, num_batches):

        """
        This function is used to replace the inactive codebook entries with the active ones, to make all codebooks
        entries to be used for training. The function has to be called periodically with the periods of "num_batches".
        For more details, the function waits for "num_batches" training batches and then discards the codebook entries
        which are used less than a specified percentage (self.discard_threshold) during this period, and replace them
        with the codebook entries which were used (active).

        Recommendation: Call this function after a specific number of training batches. In the beginning the number of
         replaced codebooks might increase. However, the main trend must be decreasing after some training time.
         If it is not the case for you, increase the "num_batches" or decrease the "discarding_threshold" to make
         the trend for number of replacements decreasing. Stop calling the function at the latest stages of training
         in order not to introduce new codebook entries which would not have the right time to be tuned and optimized
         until the end of training.

        Play with "self.discard_threshold" value and the period ("num_batches") you call the function. A common trend
        could be to select the self.discard_threshold from the range [0.01-0.1] and the num_batches from the set
        {100,500,1000,...}. For instance, as a commonly used case, if we set the self.discard_threshold=0.01 and
        num_batches=100, it means that you want to discard the codebook entries which are used less than 1 percent
        during 100 training batches. Remember you have to set the values for "self.discard_threshold" and "num_batches"
        in a logical way, such that the number of discarded codebook entries have to be in a decreasing trend during
        the training phase.

        :param num_batches: period of training batches that you want to replace inactive codebooks with the active ones

        """

        with torch.no_grad():

            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            if used_count == 0:
                print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used

                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.embedding_dim), device=self.device).clone()

            print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebooks_used[:] = 0.0


    def inference(self, input_data):

        """
        This function performs the vector quantization function for inference (evaluation) time (after training).
        This function should not be used during training.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for inference (evaluation) | shape: (NxD) )
        """

        input_data = input_data.detach().clone()
        codebooks = self.codebooks.detach().clone()
        ###########################################

        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, codebooks.t()))
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))

        min_indices = torch.argmin(distances, dim=1)
        quantized_input = codebooks[min_indices]

        #use the tensor "quantized_input" as vector quantized version of your input data for inference (evaluation) phase.
        return quantized_input


