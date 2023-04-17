import torch
import numpy as np
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

def synattack(adv_token_ids, vocab, tokenizer, adversarial_label,model, num_candidates=1):
    ## adv_token_ids shape: 10, (list)
    ## returns 10x10 numpy array 

    ## run model on adv_token padded to 512
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## outputs = []
    outputs = []
    ## for token in adv_token_ids:
    for i, token in enumerate(adv_token_ids):
        ##  replace token with unk (100)
        copied_list = np.array(adv_token_ids)
        copied_list[i] = 100
        ##  run model on new sentence padded to 512
        padded_sentence = np.pad(copied_list, (0, 512 - len(adv_token_ids)), 'constant')
        mask = torch.zeros(512, dtype=torch.int64)
        mask[0:10] = 1
        mask[i] = 0
        torch_padded_sentence =  torch.from_numpy(padded_sentence).to(dtype=torch.long)
        #inputs = torch.tensor([torch_padded_sentence, mask])
        inputs = torch.stack([torch_padded_sentence, mask], dim=0)
        inputs = torch.unsqueeze(inputs, 0).to(device)
        output = model(inputs)
        ##  append outputs to outputs[] 

        ####### TODO: VERIFY THAT OUTPUT IS (1,2) OTHERWISE WE SHOULD SQUEEZE OR SOFTMAX ON DIM 0
        sm = torch.nn.Softmax(dim=1)
        output = sm(output)
        outputs.append(output)
        ## outputs is shape 10x2

    ## pick the output with the lowest probability of adversarial label
    ## so min val of second column
    ##outputs = torch.tensor(outputs)
    outputs = torch.stack(outputs, dim=0)
    #print(outputs.shape)
    outputs = torch.squeeze(outputs)
    worst_index = torch.argmin(outputs[:,1])
    ## get synset for worst word --> pick out num_candidates that are in the vocab
    worst_word_tokenized = adv_token_ids[worst_index]
    worst_word = tokenizer.decode([100,worst_word_tokenized,101]).split(" ",1)[1].rsplit(" ", 1)[0]
    #print(worst_word)
    #worst_word.replace(" ", "")
    #worst_word_synset = wn.synonyms(worst_word) # this returns list of lists
    #concat_worst_word_synset = sum(worst_word_synset, [])
    synonyms = []
    for syn in wn.synsets(worst_word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    concat_worst_word_synset = set(synonyms)

    ## replace worst word with each synonym 
    new_candidate_tokens = []
    print(concat_worst_word_synset)
    for cand in concat_worst_word_synset:
        #print(cand)
        encoded_new_word = tokenizer.encode(cand)
        if len(encoded_new_word) > 3:
            continue
        encoded_new_word = encoded_new_word[1]
        if (encoded_new_word == worst_word_tokenized.item()):
            continue
        #print(encoded_new_word)
        if encoded_new_word in vocab:
            copied_list = np.array(adv_token_ids)
            # how does encoding single word work? I assume it's just the second index, bc you skip start token, but not sure...?
            ## yes
            copied_list[worst_index] = encoded_new_word
            new_candidate_tokens.append(copied_list)
    ## return new candidate tokens
    ## this is going to return < 10 
    #return new_candidate_tokens.detach().to(device).cpu().numpy()
    return np.array(new_candidate_tokens)
