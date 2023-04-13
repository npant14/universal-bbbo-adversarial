import torch
import numpy
from nltk.corpus import wordnet as wn

def hotflip(averaged_gradient, embedding_matrix, adv_token_ids, num_candidates=1):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_adv = torch.nn.functional.embedding(torch.IntTensor(adv_token_ids).to(device), embedding_matrix).to(device).detach().unsqueeze(0)
    averaged_gradient = averaged_gradient.unsqueeze(0)
    grad_embedding = torch.einsum("bij,kj->bik", (averaged_gradient, embedding_matrix)).to(device)
    # this is assuming average gradient is squeezed and 2 dims
    # can consider adding dim 0 if makes computation simpler elsewhere
    if num_candidates > 1:
        elts, best_k_ids = torch.topk(grad_embedding, num_candidates, dim=2)
        #print(best_k_ids)
        return best_k_ids.detach().to(device).cpu().numpy()[0]
    _, best_at_step = grad_embedding.max(2)
    return best_at_step[0].detach().to(device).cpu().numpy()
    
def bigramtokens(adv_token_ids):
    pass

def synattack(adv_token_ids, vocab, tokenizer, adversarial_label, num_candidates=1):
    ## adv_token_ids shape: 10, (list)
    ## returns 10x10 numpy array 

    ## run model on adv_token padded to 512

    ## outputs = []

    ## for token in adv_token_ids:

    ##  replace token with unk (100)

    ##  run model on new sentence padded to 512

    ##  append outputs to outputs[] 

    ## outputs is shape 10x2

    ## pick the output with the lowest probability of adversarial label

    ## get synset for worst word --> pick out num_candidates that are in the vocab

    ## replace worst word with each synonym 

    ## return new candidate tokens
