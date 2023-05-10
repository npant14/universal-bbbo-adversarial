import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentimentClassifier(torch.nn.Module):
    def __init__(self, batch_size, vocab_size, embedding_dim, embedding_weights):
        ## embedding (torch.nn.embedding)
        ## lstm (encoder)
        ## classifier (linear/ dense)
        super(SentimentClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        ## only should be not trainable if weights are passed
        # self.embedding.weight = torch.nn.Parameter(embedding_weights)
        # self.embedding.weight.requires_grad = False
        self.vocab_size = vocab_size
        self.encoder = torch.nn.LSTM(embedding_dim,
                                    hidden_size=400,
                                    num_layers=2,
                                    batch_first=True)
        self.classifier = torch.nn.Linear(in_features=512,
                                      out_features=2)

    def forward(self, tokens):
        ## tokens should be batch, [sequence, mask], 512
        ## the sequences are the actual tokenized sequences
        sequences = tokens[:, 0]

        ## the masks are 1 if real, 0 if padding
        masks = tokens[:, 1]

        ## the number of actual tokens in the sequence
        lengths = torch.sum(masks, dim=1)

        ## pack padded index expects the sequences to be sorted in increasing order
        #lengths, perm_idx = lengths.sort(0, descending=True)
        #x = sequences[perm_idx]
        
        #print(torch.min(x))
        #print(torch.max(x))
        #print(self.vocab_size)
        x = self.embedding(sequences)
        #lengths = Variable(torch.LongTensor(lengths.cpu()))
        #print(x.shape)
        #print(lengths.shape)
        x = pack_padded_sequence(x, [512]*sequences.shape[0], batch_first=True, enforce_sorted=False)
        
        x, (hn, cn) = self.encoder(x)

        x, input_sizes = pad_packed_sequence(x, batch_first=True)
        x = self.classifier(x[:, :, -1])

        # should return (batch x 2) distribution
        return x
