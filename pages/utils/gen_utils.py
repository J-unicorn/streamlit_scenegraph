
import sys
sys.path.append('/home/agens/conda_user/scene/aivg/streamlit/pages')

import os
import pickle
from contextlib import nullcontext
import torch
#import tiktoken
from torch.cuda.amp import autocast
from utils.model import GPTConfig, GPT
from transformers import PreTrainedTokenizerFast

def generation(start='test', num_samples=5, max_new_tokens=200, temperature=0.85):
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = '/home/agens/conda_user/scene/aivg/streamlit/out_t1' # ignored if init_from is not 'resume'
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else autocast(enabled=True, dtype=ptdtype)
    if init_from == 'resume':
    # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    
    model.eval()
    model.to(device)
    
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
    # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
    # enc = tiktoken.get_encoding("gpt2")
    
        enc = PreTrainedTokenizerFast.from_pretrained("gpt2")
        enc.add_special_tokens({'bos_token':'<|BOS|>', 'eos_token':'<|EOS|>', 'sep_token': '<|SEP|>', 'additional_special_tokens':['<SUBJECT>', '<OBJECT>','<PREDICATE>','<QUESTION>', '<QUESTION1>', '<QUESTION2>']})

        encode = lambda s: enc.encode(s)
        decode = lambda l: enc.decode(l, skip_special_tokens=True)
        
        # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    results = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                # front = ''
                result = decode(y[0].tolist()).split('A: ')[1].split('Q: ')[0]
                print(result)
                results.append(result)
                print('---------------')
    return results[0]
    
    
