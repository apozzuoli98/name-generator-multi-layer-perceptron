import torch
import torch.nn.functional as F
import random
import pickle
import streamlit as st


class MLPSampler:
    def __init__(self):
        with open('mlp.pkl', 'rb') as f:
            data = pickle.load(f)

        self.parameters = data['parameters']
        self.block_size = data['block_size']
        self.itos = data['itos']

        st.title("Fantasy/Sci-Fi Name Generator")
        form = st.form(key="user_settings")
        with form:
            num_input = st.number_input(label="Number of Names to Generate (Maximum: 20)", value=5, key="num_input", min_value=1, max_value=20, format='%i')
            generate_button = form.form_submit_button("Generate Names")
            if generate_button:
                st.markdown("---")
                names = self.sample(num_input, self.parameters, self.block_size)
                for name in names:
                    st.write(name)
    
    def sample(self, num_samples, parameters, block_size):
        seed = random.randint(1, 210000000)
        # seed = 42

        # Interesting seeds
        # SEED:102520266
        g = torch.Generator().manual_seed(seed)

        names = []

        C = parameters[0]
        W1 = parameters[1]
        b1 = parameters[2]
        W2 = parameters[3]
        b2 = parameters[4]

        for _ in range(num_samples):
            out = []
            context = [0] * block_size
            while True:
                emb = C[torch.tensor([context])] # (1, block_size, d)
                h = torch.tanh(emb.view(1, -1) @ W1 + b1)
                logits = h @ W2 + b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            names.append(''.join(self.itos[i] for i in out[:-1]))
            names[-1] = names[-1].capitalize()
        #     print(''.join(self.itos[i] for i in out))
        # print(f'\nSEED:{seed}')

        return names

    



if __name__ == '__main__':
    m = MLPSampler()
