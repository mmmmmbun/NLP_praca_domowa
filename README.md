# NLP_praca_domowa
Praca domowa NLP - 4 modele

Zbiór danych został pobrany z https://metatext.io/datasets/b2w-reviews01 Zawiera on 130 tysięcy rekordów z ocenami produktów e-commerce zebranymi pomiędzy styczniem a majem 2018. 

Embedding 100 wymiarów Word2Vec w portugalskim zaczerpnięty z  http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc

4 modele: LSTM, CNN, CNN z pre-trained embeddingiem I distilBERT


Model dla użytkownika do sprawdzania sentymentu, który osiągnął najlepsze wyniki w klasyfikacji: zwykłe CNN w architekturze: 

3 warstwy konwolucyjne, ze zwiększającą się liczbą filtrów (100, 125, 150) i ich rozmiarem. Warstwa dropout 0.3. Na koniec warstwa FC. Liczba słów w słowniku ograniczona do 10000.


Jak odpalić interfejs dla użytkownika:

*pobrać cały folder z onedrive (link w zadaniu)

*kliknąc plik exec model_cnn (upewnić się że wszystkie pliki sa w jednym folderze).

*wpisujemy nasz prompt (recenzja po portugalsku). Model ewaluuje sentyment recenzji. Po każdej ewaluacji program pyta użytkownika, czy zakończyć swoje działanie.



Spis plików:

s28640_praca domowa NLP - notatnik z wszystkimi modelami

model_cnn.py - skrypt - model do wywoływania 

Vocabulary.json, model.pth <- pliki do modelu cnn

model_cnn - exec file - stworzona za pomocą pyinstaller aplikacja 

