import torch

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

        self.encoder = torch.nn.LSTM(embedding_dim,
                                    hidden_size=512,
                                    num_layers=2,
                                    batch_first=True)
        self.classifier = torch.nn.Linear(in_features=512,
                                      out_features=2)
    def forward(self, tokens):
        x = self.embedding(tokens)
        x, (hn, cn) = self.encoder(x)
        x = self.classifier(x[:, -1, :])

        # should return (batch x 2) distribution
        return x