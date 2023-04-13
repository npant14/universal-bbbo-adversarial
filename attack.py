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
    outputs = []
    ## for token in adv_token_ids:
    for i, token in enumerate(adv_token_ids):
        ##  replace token with unk (100)
        copied_list = np.array(adv_token_ids)
        copied_list[i] = 100
        ##  run model on new sentence padded to 512
        padded_sentence = np.pad(copied_list, (0, 512 - len(adv_token_ids)), 'constant')
        mask = torch.zeros(512)
        mask[0:10] = 1
        mask[i] = 0
        torch_padded_sentence =  torch.from_numpy(padded_sentence)
        output = model((torch_padded_sentence, mask)
        ##  append outputs to outputs[] 

        ####### VERIFY THAT OUTPUT IS (1,2) OTHERWISE WE SHOULD SQUEEZE OR SOFTMAX ON DIM 0
        sm = nn.Softmax(dim=1)
        output = sm(output)
        outputs.append(output) ## ALSO NOT RIGHT, NEED TO SOFTMAX
        ## outputs is shape 10x2

    ## pick the output with the lowest probability of adversarial label
    ## so min val of dim 1
    outputs = np.array(outputs)
    worst_index = np.argmin(outputs, axis=1)
    ## get synset for worst word --> pick out num_candidates that are in the vocab
    worst_word_tokenized = adv_token_ids[worst_index]
    worst_word = tokenizer.decode(worst_word_tokenized)
    worst_word_synset = wn.synonyms(worst_word) # this returns list of lists
    concat_worst_word_synset = sum(worst_word_synset, [])
    ## replace worst word with each synonym 
    new_candidate_tokens = []
    for cand in concat_worst_word_synset:
        encoded_new_word = encoder.encode(cand)[1]
        if encoded_new_word in vocab:
            copied_list = np.array(adv_token_ids)
            # how does encoding single word work? I assume it's just the second index, bc you skip start token, but not sure...?
            copied_list[worst_index] = encoded_new_word
            new_candidate_tokens.append(copied_list)
    ## return new candidate tokens
    return np.array(new_candidate_tokens)
