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
    # load the binary SST dataset.
    # single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    #reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
    #                                                token_indexers={"tokens": single_id_indexer},
    #                                                use_subtrees=True)
    #train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    # reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
    #                                                 token_indexers={"tokens": single_id_indexer})
    # dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    # test_dataset = reader.read('data/sst/test.txt')

    #vocab = Vocabulary.from_instances(train_data)

    train_dataset = torchtext.datasets.SST2(split = 'train')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tokenized_train2 = []

    ### UNCOMMENT BELOW TO TOKENIZE FROM SCRATCH
    #for i,s in enumerate(train_dataset):
    #    print(i)
    #    tokenized_train2.append((tokenizing_sst2(s[0]), s[1]))
    #    if i == 20:
    #        break

    #tokenized = torch.cat(list(zip(*tokenized_train2))[0])
    #np.save("tokenized_train.npy", np.asarray(tokenized))
    #np.save("train_labels.npy", np.asarray(list(list(zip(*tokenized_train))[1])))

    ### UNCOMMENT BELOW TO LOAD IN DATA FROM FILES
    tokenized_train = []
    training_data = np.load("tokenized_train.npy")
    training_labels = np.load("train_labels.npy")
    for i in range(training_labels.shape[0]):
        tokenized_train.append((torch.tensor(training_data[i*512:512*(i+1)]).to(device), torch.tensor(training_labels[i]).to(device)))
    
    #print(tokenized_train[0])
    #print(tokenized_train2[0])
    #print(len(tokenized_train))
    #print(training_data.shape)
    #print(training_labels.shape)
    #return
    trainloader = DataLoader(tokenized_train, batch_size = 512)
    
    # val_dataset = torchtext.datasets.SST2(split = 'dev')
    # tokenized_val  = [tokenizer.tokenize(s) for s in val_dataset]
    # valloader = DataLoader(tokenized_val, batch_size = 1024)


    # ## get pretrained word2vec embedding weights
    # embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
    # embedding_weights = _read_pretrained_embeddings_file(embedding_path,
    #                                             embedding_dim=300,
    #                                             vocab=vocab,
    #                                             namespace="tokens")
    embedding_weights = None
    batch_size = 1
    vocab_size, max_token = get_vocab_size(trainloader)
    print(f"vocab size = {max_token}")
    embedding_dim = 300
    model = SentimentClassifier(batch_size, max_token, embedding_dim, embedding_weights)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()


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

if __name__ == '__main__':
    main()
