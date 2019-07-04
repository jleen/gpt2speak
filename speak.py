'''
    jleen
    based upon code by TaeHwan Jung(@graykode)
'''
import comtypes.client
import numpy
import os
import random
import sys
import torch

from GPT2.config import GPT2Config
from GPT2.encoder import get_encoder
from GPT2.model import (GPT2LMHeadModel)
from GPT2.sample import sample_sequence
from GPT2.utils import load_weight

SpVoice = comtypes.client.CreateObject('SAPI.SpVoice')

SVSFlagsAsync = 1


def text_generator(state_dict):
    batch_size = 1
    nsamples=1
    assert nsamples % batch_size == 0

    seed = random.randint(0, 2147483647)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    length = 10
    # length = config.n_ctx // 2
    if length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    quiet = False
    temperature = 0.7
    top_k = 40
    text = ''
    unconditional = True
    print(text)
    context_tokens = enc.encode(text)

    while True:
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sample_sequence(
                model=model, length=length,
                context=context_tokens  if not unconditional else None,
                start_token=enc.encoder['<|endoftext|>'] if unconditional else None,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, device=device
            )
            out = out[:, len(context_tokens):].tolist()
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                if quiet is False:
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
                SpVoice.WaitUntilDone(-1)
                SpVoice.Speak(text, SVSFlagsAsync)


def speak():
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        text_generator(state_dict)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()

