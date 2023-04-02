from models import SentimentClassifier
#from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
        #    StanfordSentimentTreeBankDatasetReader
#from allennlp.data.token_indexers import SingleIdTokenIndexer
#from allennlp.data.vocabulary import Vocabulary
import torchtext
from torch.utils.data import DataLoader
from torch import optim
import torch
from transformers import BertTokenizer
import numpy as np
from attack import hotflip

def get_vocab_size(dataloader):
    vocab = set()
    for batch in dataloader:
        inputs, labels = batch
        #print(inputs.shape)
        #print(labels.shape)
        for sequence in inputs:
            for word in sequence:
                vocab.add(word.item())
    print(vocab)
    return len(vocab), max(vocab) + 1
    
def tokenizing_sst2(sentence):
    input_ids = []
    attention_mask = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_sentence = tokenizer.encode_plus(sentence,
                                        add_special_tokens=True,
                                        max_length=512,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        return_tensors='pt')
    #input_ids.append(tokenized_sentence['input_ids'])
    #attention_mask.append(tokenized_sentence['attention_mask'])
    return torch.squeeze(tokenized_sentence['input_ids'])
    #return {'input_ids': torch.cat(input_ids, dim=0), 'attention_mask': torch.cat(attention_mask, dim=0)}
    
    
def main():

    training_model = False

    train_dataset = torchtext.datasets.SST2(split = 'train')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### UNCOMMENT BELOW TO TOKENIZE FROM SCRATCH
    #tokenized_train = []
    #for i,s in enumerate(train_dataset):
    #    print(i)
    #    tokenized_train.append((tokenizing_sst2(s[0]), s[1]))
    #    if i == 20:
    #        break

    #tokenized = torch.cat(list(zip(*tokenized_train))[0])
    #np.save("tokenized_train.npy", np.asarray(tokenized))
    #np.save("train_labels.npy", np.asarray(list(list(zip(*tokenized_train))[1])))

    ### UNCOMMENT BELOW TO LOAD IN DATA FROM FILES
    tokenized_train = []
    training_data = np.load("tokenized_train.npy")
    training_labels = np.load("train_labels.npy")
    for i in range(training_labels.shape[0]):
        tokenized_train.append((torch.tensor(training_data[i*512:512*(i+1)]).to(device), torch.tensor(training_labels[i]).to(device)))
    
    trainloader = DataLoader(tokenized_train, batch_size = 512)
    
    # val_dataset = torchtext.datasets.SST2(split = 'dev')
    # tokenized_val  = [tokenizer.tokenize(s) for s in val_dataset]
    # valloader = DataLoader(tokenized_val, batch_size = 1)

    embedding_weights = None
    batch_size = 1
    vocab_size, max_token = get_vocab_size(trainloader)
    print(f"vocab size = {max_token}")
    embedding_dim = 300
    model = SentimentClassifier(batch_size, max_token, embedding_dim, embedding_weights)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    if training_model: 
        optimizer = optim.Adam(model.parameters())
        for epoch in range(10):
            total_loss = 0
            count = 0

            for i, batch in enumerate(trainloader):
                count += 1
                inputs, labels = batch
                inputs.to(device)
                labels.to(device)
                preds = model(inputs)
                loss = loss_fn(preds, labels)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            print(f'Training Loss at epoch {epoch} : {total_loss / count}')
            torch.save(model.state_dict(), "weights.npy")
    else:
        model.load_state_dict(torch.load("weights.npy",map_location=device))
    
    model.eval()

    extracted_grads = []
    def extract_grad_hook(moddule, grad_in, grad_out):
        extracted_grads.append(grad_out[0])

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            embedding_matrix = module.weight.cpu().detach()
            module.weight.requires_grad = True
            module.register_full_backward_hook(extract_grad_hook)

    universal_perturb_batch_size = 512
    



    ###
    trainloader_fk = DataLoader(tokenized_train, batch_size = 1)
    postive_val_target = []
    ## batch size 1
    for i, batch in enumerate(trainloader_fk):
        inputs, labels = batch
        if labels[0] == 0:
            ## append tuple of inputs labels
            postive_val_target.append(batch)

    ## get acc

    model.train()

    ## TODO: initialize which trigger token IDs to use
    trigger_token_ids = [1]
    target_label = 1

    for i, batch in enumerate(postive_val_target):
        inputs, labels = batch
        inputs.to(device)
        labels.to(device)

        dummy_optimizer = optim.Adam(model.parameters())
        dummy_optimizer.zero_grad()

        original_labels = labels[0].clone()
        label = torch.IntTensor(target_label)

        extracted_grads = []
        preds = model(inputs)
        loss = loss_fn(preds, labels)
        loss.backward()

        grads = extracted_grads[0].cpu()

        label = original_labels
        average_grad = torch.sum(grads, dim=0)
        average_grad = average_grad[0:len(trigger_token_ids)]

        candidate_trigger_token_ids = hotflip(average_grad, embedding_matrix, trigger_token_ids)



if __name__ == '__main__':
    main()
