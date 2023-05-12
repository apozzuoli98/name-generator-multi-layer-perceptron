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

class MLP:
    def __init__(self):
        # random.seed = (42) # used for debugging
        # torch.manual_seed(42) # for reproducibility

        save_model = True # Set to true if you want the model saved 


        self.words = open('names.txt', 'r').read().splitlines()
        random.shuffle(self.words)

        self.context = 8 # context length (block size) where n is number of characters used to predict next char
        
        # build the vocabulary of characters mapping to integers
        self.chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}
        vocab_size = len(self.itos)

        # build training, dev, and test datasets
        n1 = int(0.8*len(self.words))
        n2 = int(0.9*len(self.words))
        self.Xtr, self.Ytr = self.build_dataset(self.words[:n1], self.context) # 80%
        self.Xdev, self.Ydev = self.build_dataset(self.words[n1:n2], self.context) # 10%
        self.Xte, self.Yte = self.build_dataset(self.words[n2:], self.context) # 10%

        # initialize parameters
        self.model = self.init_params(vocab_size)

        # train the mlp
        self.train(self.model)

        # put layers into eval mode (needed for batchnorm)
        for layer in self.model.layers:
            layer.training = False
        
        # evaluate the loss
        self.split_loss('train', self.model)
        self.split_loss('val', self.model)

        # sample from the model
        # self.sample(self.model)

        # Save the mlp
        if save_model:
            save_data = {
                'model': self.model,
                'block_size': self.context,
                'itos': self.itos
            }
            with open('mlp.pkl', 'wb') as f:
                pickle.dump(save_data, f)


    def build_dataset(self, words, block_size):
        """
        Build the dataset such that block_size number of characters are used to 
        predict next character

        :words: all words in list
        :block_size: number of characters used to predict next char

        :return: X, Y context and target char
        """
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y
    


    def init_params(self, vocab_size):
        """
        Build the model
        """
        n_embd = 10
        n_hidden = 16

        model = Sequential([
            Embedding(vocab_size, n_embd),
            FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, vocab_size)
        ])

        # parameter init
        with torch.no_grad():
            model.layers[-1].weight *= 0.1 # make last layer less confident

        return model
    

    def train(self, model):
        """
        Train the model
        """
        parameters = model.parameters()
        for p in parameters:
            p.requires_grad = True

        max_steps = 200000
        batch_size = 32
        lossi = []

        for i in range(max_steps):

            # minibatch construct
            ix = torch.randint(0, self.Xtr.shape[0], (batch_size,))
            Xb, Yb = self.Xtr[ix], self.Ytr[ix] # batch X, Y

            # forward pass
            logits = model(Xb)
            loss = F.cross_entropy(logits, Yb) # loss function

            # backward pass
            for p in parameters:
                p.grad = None
            loss.backward()

            # update: simple SGD
            lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
            for p in parameters:
                p.data += -lr * p.grad

            # track stats
            if i % 10000 == 0: # print every once in a while
                print(f'{i:7d}/{max_steps}: {loss.item():.4f}')
            lossi.append(loss.log10().item())

        # plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
        # plt.show()

    @torch.no_grad() # this decorator disables gradient tracking inside pytorch
    def split_loss(self, split, model):
        """
        Evaluate the loss
        """
        x,y = {
            'train': (self.Xtr, self.Ytr),
            'val': (self.Xdev, self.Ydev),
            'test': (self.Xte, self.Yte)
        }[split]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        print(split, loss.item())

    def sample(self, model):
        """
        Sample from the model
        """
        for _ in range(20):

            out = []
            context = [0] * self.context # initialize with all ...
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

            print(''.join(self.itos[i] for i in out)) # decode and print the generated word


# --------------------------------------------------

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

        for _ in range(num_samples):
            out = []
            context = [0] * block_size # initialize with all ...
            while True:
                # forward pass the neural net
                x = torch.tensor([context])
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
