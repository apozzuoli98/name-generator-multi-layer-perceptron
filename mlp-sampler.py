import torch
import torch.nn.functional as F
import random
# import pickle
import dill as pickle
from dill import dumps, loads
import streamlit as st


# ----------------------------------------------------------------
class Linear:
    """
    Linear layer building block
    """
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

# ----------------------------------------------------------------------
class BatchNorm1d:
    """
    Batch normalization layer building block
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True) # batch mean
            xvar = x.var(dim, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
# -------------------------------------------------------------------------
class Tanh:
    """
    Tanh activation layer
    """
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

# -----------------------------------------------------------------------
class Embedding:
    """
    Embedding layer
    """
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
    
# ----------------------------------------------------------------------
class FlattenConsecutive:
    """
    Flatten layer
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view (B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
# ------------------------------------------------------------
class Sequential:
    """
    Call all layers in sequence
    """
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        # get parameters of all layers and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]
    

# ---------------------------------------------------

class MLPSampler:
    def __init__(self):
        """Samples from a trained MLP and displays the results"""

        # load the mlp
        with open('mlp-new.pkl', 'rb') as f:
            data = pickle.load(f)


        self.model = data['model']
        self.block_size = data['block_size']
        self.itos = data['itos']

        # display form in streamlit
        st.title("Fantasy/Sci-Fi Name Generator")
        form = st.form(key="user_settings")
        with form:
            num_input = st.number_input(label="Number of Names to Generate (Maximum: 20)", value=5, key="num_input", min_value=1, max_value=20, format='%i')
            generate_button = form.form_submit_button("Generate Names")
            if generate_button:
                st.markdown("---")
                # sample names
                names = self.sample(num_input, self.model, self.block_size)
                for name in names:
                    st.write(name)


    def sample(self, num_samples, model, block_size):
        """
        Sample from the model
        """
        seed = random.randint(1, 210000000)
        torch.manual_seed(seed)
        names = []
        
        st.write(model(torch.tensor([context])))

        for _ in range(num_samples):
            out = []
            context = [0] * block_size # initialize with all ...
            while True:
                # forward pass the neural net
                logits = model(torch.tensor([context])) # embed the characters
                probs = F.softmax(logits, dim=1)
                # sample from the distribution
                ix = torch.multinomial(probs, num_samples=1).item()
                # shift the context window and track the samples
                context = context[1:] + [ix]
                out.append(ix)
                # if we sample the special '.' token, break
                if ix == 0:
                    break
            names.append(''.join(self.itos[i] for i in out[:-1]))
            names[-1] = names[-1].capitalize()
        return names


if __name__ == '__main__':
    m = MLPSampler()
