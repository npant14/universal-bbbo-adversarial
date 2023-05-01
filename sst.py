from models import SentimentClassifier
from ngrams import basic_clean, get_data, get_next_words
from operator import itemgetter
import heapq
from copy import deepcopy
import torchtext
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import optim
import torch
from transformers import BertTokenizer
import numpy as np
from attack import hotflip, synattack
import pickle
import csv
import random
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


#def get_vocab_size(dataloader):
#    vocab = set()
#    vocab_map = {}
#    for batch in dataloader:
#        inputs, labels = batch
#        #print(inputs.shape)
#        #print(labels.shape)
#        for sequence in inputs[:,0]:
#            for word in sequence:
#                vocab.add(word.item())
#    # print(vocab)
#    vocab = list(vocab)
#    for i, word in enumerate(vocab):
#        vocab_map[word] = i
#    with open('vocab_dump.pickle', 'wb') as file:
#        pickle.dump({"vocab_len": len(vocab), "max_token": max(vocab) + 1, "vocab": vocab, "vocab_map": vocab_map}, file, protocol=pickle.HIGHEST_PROTOCOL)
#    return len(vocab), max(vocab) + 1, vocab, vocab_map
    
def initialize_tokens(initial_word, length):
    
    tb = get_data("books_1.Best_Books_Ever.csv")
    bigram_dict = get_next_words(tb)
    words = [initial_word]
    for i in range(1, length):
        if words[-1] in bigram_dict:
            words.append(bigram_dict[words[-1]][0])
        else:
            random_num = random.randint(0,len(bigram_dict.keys()) - 1)
            words.append(list(bigram_dict)[random_num])
    return words

def tokenize_word(word, token_dict, untoken_dict):
    if word not in token_dict:
        token = len(token_dict) + 2 ## put in a +2 here to account for start and stop tokens
        token_dict[word] = token
        untoken_dict[token] = word
    return token_dict[word]

def tokenizing_sst2(sentence, token_dict, untoken_dict):
    input_ids = []
    attention_mask = []
    #print(sentence)
    sentence = basic_clean(sentence)
    #print(sentence)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_sentence = [0]
    
    for word in sentence:
        token = tokenize_word(word, token_dict, untoken_dict)
        tokenized_sentence.append(token)

    tokenized_sentence.append(1)
    tokenized_sentence  = tokenized_sentence +  [0] * (512 - len(tokenized_sentence))
    attention_mask = [1] * (len(tokenized_sentence)) + [0] * (512 - len(tokenized_sentence))

    return torch.stack([torch.tensor(tokenized_sentence),
                        torch.tensor(attention_mask)], dim=0)
    

def get_accuracy(model, device, dataloader, trigger_token_ids=None):
    model.eval()
    
    if trigger_token_ids is None:
        total_examples = 0
        total_correct = 0
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)

            # Total number of labels
            total_examples += labels.size(0)

            # Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)

            # Calculate the number of correct answers
            correct = (predicted == labels).sum().item()

            total_correct += correct

    else: ## trigger_token_ids is not None
        total_examples = 0
        total_correct = 0
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            trigger_sequence_tensor = torch.tensor(trigger_token_ids, dtype=torch.int64)
            trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch) - 1, 1).to(device)
            # print(trigger_sequence_tensor.shape)
            # print(inputs[:,0].shape)
            altered_sequences = torch.cat((trigger_sequence_tensor, inputs[:,0]), 1)[:,:512]
            altered_inputs = torch.stack((altered_sequences, inputs[:,1]), dim=1)
            outputs = model(altered_inputs)

            # Total number of labels
            total_examples += labels.size(0)

            # Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)
            #print(outputs, predicted)
            # Calculate the number of correct answers
            correct = (predicted == labels).sum().item()

            total_correct += correct

    acc = (total_correct / total_examples) * 100
    return acc

def get_loss(model, device, dataloader, trigger_token_ids=None):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    if trigger_token_ids is None:
        total_examples = 0
        total_correct = 0
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(preds, labels)

            # Total number of labels
            total_examples += labels.size(0)

            total_loss += loss.item()

    else: ## trigger_token_ids is not None
        total_examples = 0
        total_loss = 0
        for batch in dataloader:
            #print(batch)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            trigger_sequence_tensor = torch.tensor(trigger_token_ids, dtype=torch.int64)
            trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(batch) - 1, 1).to(device)
            #print(trigger_sequence_tensor.shape)
            #print(inputs[:,0].shape)
            inputs = inputs.unsqueeze(0)
            #print(inputs.shape)
            #print(trigger_sequence_tensor.shape)
            altered_sequences = torch.cat((trigger_sequence_tensor, inputs[:,0]), 1)[:,:512]
            altered_inputs = torch.stack((altered_sequences, inputs[:,1]), dim=1)
            outputs = model(altered_inputs)
            loss = loss_fn(outputs, labels)

            # Total number of labels
            total_examples += labels.size(0)

            total_loss += loss.item()

    loss = total_loss / total_examples
    # print(loss)
    return loss

def get_best_candidates(model, batch, device, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
    """
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    ## maintain a heapq

    ## run on index 0 for candidates
    loss_per_candidate = get_loss_per_candidate(0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids, device)

    ## this is a list that is (beam size) long sorted by maximum loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    ## for len trigger tokens (1-end)
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
    ##  for everything in the heapq
        for candidate, cand_loss in top_candidates:
    ##      collect list of losses at index
            loss_per_candidate.extend(get_loss_per_candidate(idx, model, batch, candidate,
                                                             cand_trigger_token_ids, device))
    ##      update heapq with new top n tokens
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    ## return final top n tokens
    return max(top_candidates, key=itemgetter(1))[0]

def get_loss_per_candidate(index, model, batch, trigger_token_ids, cand_trigger_token_ids, device):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    ## evaluate loss on current set of best trigger token ids
    loss_per_candidate = []

    # get_loss(model, device, dataloader, trigger_token_ids=None):
    #print(batch[0][0].shape)
    #print(batch[1])

    cur_loss = get_loss(model, device, [(batch[0][0], batch[1])], trigger_token_ids)
    loss_per_candidate.append((deepcopy(trigger_token_ids), cur_loss))

    ## iterate through set of candidate tokens at that index replacing one at a time, at index, and save loss
    for candidate in cand_trigger_token_ids[index]:
        triggers_one_replaced = deepcopy(trigger_token_ids)
        triggers_one_replaced[index] = candidate
        loss = get_loss(model, device, [(batch[0][0], batch[1])], triggers_one_replaced)
        loss_per_candidate.append((deepcopy(triggers_one_replaced), loss))

    ## expected return in code we are copying is 
    ## list(tuple(list of tokens, loss)) 
    return loss_per_candidate

def main():

    training_model = True
    tokenize_from_scratch = True

    train_dataset = torchtext.datasets.SST2(split = 'train')
    val_dataset = torchtext.datasets.SST2(split = 'dev')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    token_dict = {}
    untoken_dict = {0 : "STK", 1 : "ETK"}

    if tokenize_from_scratch:
        tokenized_train = []
        for i,s in enumerate(train_dataset):
            # now has both sequence and mask
            print(i)
            tokenized_train.append((tokenizing_sst2(s[0], token_dict, untoken_dict), s[1]))


        tokenized = torch.cat(list(zip(*tokenized_train))[0])
        np.save("tokenized_train.npy", np.asarray(tokenized))
        np.save("train_labels.npy", np.asarray(list(list(zip(*tokenized_train))[1])))


        tokenized_val = []
        for i, s in enumerate(val_dataset):
        #    print(i)
           tokenized_val.append((tokenizing_sst2(s[0], token_dict, untoken_dict), s[1]))
        tokenized = torch.cat(list(zip(*tokenized_val))[0])
        np.save("tokenized_val.npy", np.asarray(tokenized))
        np.save("val_labels.npy", np.asarray(list(list(zip(*tokenized_val))[1])))

        ## also tokenize words from ngrams

        tb = get_data("books_1.Best_Books_Ever.csv")
        bigram_dict = get_next_words(tb)
        for word in bigram_dict.keys():
            tokenize_word(word, token_dict, untoken_dict)
        for wordlist in bigram_dict.values():
            for word in wordlist:
                tokenize_word(word, token_dict, untoken_dict)

        ## pickle dict
        with open('token_dicts.pkl', 'wb') as f:
            pickle.dump((token_dict, untoken_dict), f)

    else: # not tokenize_from_scratch
        tokenized_train = []
        training_data = np.load("tokenized_train.npy")
        training_labels = np.load("train_labels.npy")
        for i in range(training_labels.shape[0]):
            tokenized_train.append((torch.tensor(training_data[i*2:2*(i+1)]).to(device), torch.tensor(training_labels[i]).to(device)))

        tokenized_val = []
        val_data = np.load("tokenized_val.npy")
        val_labels = np.load("val_labels.npy")
        for i in range(val_labels.shape[0]):
            tokenized_val.append((torch.tensor(val_data[i*2:2*(i+1)]).to(device), torch.tensor(val_labels[i]).to(device)))
        
        with open('token_dicts.pkl', 'rb') as f:
            token_dict, untoken_dict = pickle.load(f)

    trainloader = DataLoader(tokenized_train, batch_size = 256)
    valloader = DataLoader(tokenized_val, batch_size=1)
    embedding_weights = None
    batch_size = 1
    
    #if tokenize_from_scratch:
    #    vocab_size, max_token, vocab, vocab_map = get_vocab_size(trainloader) 
    #else:
    #    with open("vocab_dump.pickle", "rb") as file:
    #        loaded = pickle.load(file)
    #        vocab_size, max_token, vocab, vocab_map = loaded["vocab_len"], loaded["max_token"], loaded["vocab"], loaded["vocab_map"]
    max_token = len(token_dict)+2
    print(f"vocab size = {max_token}")
    embedding_dim = 300
    model = SentimentClassifier(batch_size, max_token, embedding_dim, embedding_weights)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    if training_model: 
        print("entering training loop")
        optimizer = optim.Adam(model.parameters())
        for epoch in range(1):
            total_loss = 0
            count = 0

            for i, batch in enumerate(trainloader):
                count += 1
                inputs, labels = batch
                print(inputs.shape)
                print(labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = loss_fn(preds, labels)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            print(f'Training Loss at epoch {epoch} : {total_loss / count}')
            torch.save(model.state_dict(), "weights.h5")
    else:
        model.load_state_dict(torch.load("weights.h5",map_location=device))
    
    model.eval()

    extracted_grads = []
    def extract_grad_hook(moddule, grad_in, grad_out):
        extracted_grads.append(grad_out[0])

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            embedding_matrix = module.weight.to(device).detach()
            module.weight.requires_grad = True
            module.register_full_backward_hook(extract_grad_hook)

    universal_perturb_batch_size = 512


    ###
    positive_val_target = []
    ## batch size 1
    for i, batch in enumerate(valloader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        if labels[0] == 0:
            ## append tuple of inputs labels
            positive_val_target.append((inputs, labels))

    ## get acc
    print("initial training accuracy")
    #print(get_accuracy(model, device, trainloader))
    print("initial val accuracy")
    #print(get_accuracy(model, device, positive_val_target))

    model.train()

    ## TODO: initialize which trigger token IDs to use

    ## start with 10 random words here (not just always racist)
    #trigger_token_ids = [tokenizing_sst2("black")[0][1]] * 10
    initial_word = "black"
    trigger_len = 5
    trigger_token_ids = initialize_tokens(initial_word, trigger_len)
    print(trigger_token_ids)

    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trigger_token_ids = [token_dict[word] for word in trigger_token_ids]
    print(trigger_token_ids)
    print(type(trigger_token_ids))
    print(len(trigger_token_ids))
    #trigger_token_ids = trigger_token_ids[:,1:-1]
    #for i, lst in enumerate(trigger_token_ids):
    #    trigger_token_ids[i] = lst[1:-1]
    #trigger_token_ids = sum(trigger_token_ids, [])
    print(trigger_token_ids)
    
    target_label = 1

    # print(f"positive val loader loop size {len(positive_val_target)}")

    for i, batch in enumerate(positive_val_target):
        #print(i, len(positive_val_target))
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

        ## want to replace this 
        ## 10x10 matrix
        ## in general (num candidates x num tokens in trigger string)

        # candidate_trigger_token_ids = hotflip(average_grad, embedding_matrix, trigger_token_ids, num_candidates=10)
        #print(trigger_token_ids)
        vocab = None
        candidate_trigger_token_ids = synattack(trigger_token_ids, vocab, token_dict, untoken_dict, target_label, model, num_candidates=10)
        print(f"CANDIDATE TRIGGER TOKEN IDS BEFORE GETBESTCAND {candidate_trigger_token_ids}")
        trigger_token_ids = get_best_candidates(model, batch, device, trigger_token_ids, candidate_trigger_token_ids, beam_size=1)

        best_token_acc = get_accuracy(model, device, positive_val_target, trigger_token_ids)

        print(f"accuracy on round {i}: {best_token_acc}")

        f = open('output_all.csv', 'a')
        # create the csv writer
        writer = csv.writer(f)
        # write candidate trigger tokens to the csv
        #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        for tti in trigger_token_ids:
            writer.writerow([untoken_dict[idx.item()] for idx in tti] + [str(best_token_acc)])
            writer.writerow([])

        ## for random reinitialization of the tokens
        if random.uniform(0, 1) < 0.1:
            initial_word = random.choice(list(token_dict.keys()))
            new_random_token_ids = initialize_tokens(initial_word, trigger_len)
            new_random_token_ids = [token_dict[word] for word in new_random_token_ids]
            new_random_token_ids = get_best_candidates(model, batch, device, trigger_token_ids, new_random_token_ids, beam_size=1)

            if best_token_acc >= get_accuracy(model, device, positive_val_target, new_random_token_ids):
                trigger_token_ids = new_random_token_ids

        # get_accuracy(model, device, positive_val_target, trigger_token_ids)
        # break

    # open the file in the write mode
    f = open('output.csv', 'w')
    # create the csv writer
    writer = csv.writer(f)
    # write candidate trigger tokens to the csv
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for candidate in candidate_trigger_token_ids:
        writer.writerow([untoken_dict[idx.item()] for idx in candidate])
        writer.writerow([])


if __name__ == '__main__':
    main()
