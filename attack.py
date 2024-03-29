import torch
import random
import numpy as np
from nltk.corpus import wordnet as wn
from preprocess import tokenize_word, initialize_trigger_words


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

def synattack(adv_token_ids, token_dict, untoken_dict, adversarial_label, model, num_candidates=1):
    """
    adv_token_ids : 
    """
    ## adv_token_ids shape: 10, (list)

    ## returns (10,10) numpy array 

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
        mask[0:len(adv_token_ids)] = 1
        ## 0 out attention mask for unk token
        mask[i] = 0
        torch_padded_sentence =  torch.from_numpy(padded_sentence).to(dtype=torch.long)
        inputs = torch.stack([torch_padded_sentence, mask], dim=0)
        inputs = torch.unsqueeze(inputs, 0).to(device)
        output = model(inputs)
        ##  append outputs to outputs[] 

        sm = torch.nn.Softmax(dim=1)
        output = sm(output)
        outputs.append(output)
        ## outputs is shape 10x2

    ## pick the output with the lowest probability of adversarial label
    ## so min val of second column
    outputs = torch.stack(outputs, dim=0)
    outputs = torch.squeeze(outputs)
    worst_index = torch.argmin(outputs[:,adversarial_label])
    ## get synset for worst word --> pick out num_candidates that are in the vocab
    worst_word_tokenized = adv_token_ids[worst_index]
    worst_word = untoken_dict[worst_word_tokenized]
    print(worst_word)
    synonyms = []
    for syn in wn.synsets(worst_word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    concat_worst_word_synset = set(synonyms)
    
    
    if len(concat_worst_word_synset) == 0:
        print("reinitializing in attack.py :) have a great day !")
        new_cand_tokens = []
        for i in range(5):
            initial_word = random.choice(list(token_dict.keys()))
            initial_words = initialize_trigger_words(initial_word, 5)
            tokenized = [token_dict[word] for word in initial_words]
            new_cand_tokens.append(np.array(tokenized))
        return np.array(new_cand_tokens)
    ## replace worst word with each synonym 
    new_candidate_tokens = []
    print(concat_worst_word_synset)
    for cand in concat_worst_word_synset:
        if (cand == worst_word):
            continue
        if cand in token_dict:
            encoded_new_word = tokenize_word(cand, token_dict, untoken_dict)
            copied_list = np.array(adv_token_ids)
            copied_list[worst_index] = encoded_new_word
            new_candidate_tokens.append(copied_list)
    ## return new candidate tokens
    ## this is going to return < 10 
    #return new_candidate_tokens.detach().to(device).cpu().numpy()
    return np.array(new_candidate_tokens)


def run_hotflip_attack(model, batch, device, loss_fn, embedding_matrix):
    
    extracted_grads = []
    def extract_grad_hook(moddule, grad_in, grad_out):
        extracted_grads.append(grad_out[0])

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            embedding_matrix = module.weight.to(device).detach()
            module.weight.requires_grad = True
            module.register_full_backward_hook(extract_grad_hook)

    universal_perturb_batch_size = 512

    model.train()
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    dummy_optimizer = optim.Adam(model.parameters())
    dummy_optimizer.zero_grad()

    original_labels = labels[0].clone().to(device)
    label = torch.IntTensor(target_label).to(device)

    extracted_grads = []
    print(inputs)
    preds = model(inputs)
    loss = loss_fn(preds, labels)
    loss.backward()

    grads = extracted_grads[0].to(device)

    label = original_labels.to(device)
    average_grad = torch.sum(grads, dim=0).to(device)
    average_grad = average_grad[0:len(trigger_token_ids)]

    candidate_trigger_token_ids = hotflip(average_grad, embedding_matrix, trigger_token_ids, num_candidates=10)

    return candidate_trigger_token_ids

