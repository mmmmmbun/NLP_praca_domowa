import torch
import numpy as np
import json
import torch.nn as nn
import torch.optim as optim
import re
import os
import sys
 
script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(script_path)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, filter_sizes, num_filters_per_size):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, 
                       out_channels=num_filters, 
                       kernel_size=f_size) for f_size, num_filters in zip(filter_sizes, num_filters_per_size)]
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(num_filters_per_size), num_classes)


    def forward(self, x):
        x = self.embedding(x)  # x: [batch_size, sequence_length, embedding_dim]
        x = x.permute(0, 2, 1)  # x: [batch_size, embedding_dim, sequence_length]

        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [torch.max_pool1d(c, c.shape[2]).squeeze(2) for c in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        return self.fc(x)


def check_and_preprocess_string(sentence):
    # Sprawdź czy jest spam (powtarzające sie sekwencje conajmniej kilkukrotnie)
    if re.search(r'(\b[\w\s,.!?;:]{5,}\b)(\1{2,})', sentence):
        return None 
    sentence = re.sub(r"[^\w\s]", '', sentence)  
    sentence = re.sub(r"\s+", ' ', sentence)     
    sentence = re.sub(r"\d", '', sentence)
    return sentence


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# Słownik
with open('vocabulary.json', 'r') as vocab_file:
    vocab = json.load(vocab_file)
vocab_size = len(vocab)  


embedding_dim = 256  # Wymiar embeddingów
num_classes =  2  # Liczba klas
filter_sizes = [3, 4, 5]  # Rozmiary filtrów
num_filters = [100, 125, 150]  # Liczba filtrów

model = TextCNN(vocab_size, embedding_dim, num_classes, filter_sizes, num_filters)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

device = torch.device("cpu")

model.load_state_dict(torch.load('model.pth'))
model.eval()


def predict_text_CNN(text):
    word_seq = np.array([vocab.get(check_and_preprocess_string(word), 0) for word in text.split()])
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding_(word_seq, 500))
    inputs = pad.to(torch.device('cpu'))

    with torch.no_grad():
        output = model(inputs)
        prediction = output.argmax(dim=1).item()
        if prediction == 1:
            return  'Ta recenzja jest pozytywna'
        else:
            return 'Ta recenzja jest negatywna'


if __name__ == "__main__":
    while True:
        user_input = input("Wpisz recenzje po portugalsku, żeby przeprowadzić analizę sentymentu: ")
        result = predict_text_CNN(user_input)
        print(result)

        next_evaluation = input("Czy chcesz przeprowadzić kolejną analizę? (tak/nie): ")
        if next_evaluation.lower() != 'tak':
            print("Kończenie programu.")
            break