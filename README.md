# NLP_praca_domowa
Praca domowa NLP - 4 modele
Zbiór danych został pobrany z https://metatext.io/datasets/b2w-reviews01 Zawiera on 130 tysięcy rekordów z ocenami produktów e-commerce zebranymi pomiędzy styczniem a majem 2018. 

Embedding 100 wymiarów Word2Vec w portugalskim zaczerpnięty z  http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc

4 modele: LSTM, CNN, CNN z pre-trained embeddingiem I distilBERT

Model dla użytkownika do sprawdzania sentymentu, który osiągnął najlepsze wyniki w klasyfikacji: zwykłe CNN w architekturze: TextCNN(
  (embedding): Embedding(10000, 256)
  (convs): ModuleList(
    (0): Conv1d(256, 100, kernel_size=(3,), stride=(1,))
    (1): Conv1d(256, 125, kernel_size=(4,), stride=(1,))
    (2): Conv1d(256, 150, kernel_size=(5,), stride=(1,))
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=375, out_features=2, bias=True)
)

3 warstwy konwolucyjne, ze zwiększającą się liczbą filtrów (100, 125, 150) i ich rozmiarem. Warstwa dropout 0.5. Na koniec warstwa FC. Liczba słów w słowniku ograniczona do 10000.

Jak odpalić interfejs dla użytkownika:
Odpalamy plik przez terminal
python3 cnn.app
I wpisujemy nasz prompt (recenzja po portugalsku)

Spis plików:
s28640_praca domowa NLP - notatnik z wszystkimi modelami
model_cnn.py - model do wywoływania dla użytkownika
Vocabulary.json, model.pth <- pliki do modelu dla użytkowniku
