
# projekt_studencki

Projekt studencki - fizyka. Celem projektu jest analiza danych z symulacji kalorymetru za pomocą metod uczenia maszynowego (3D CNN).

* dane_info.txt - informacja o surowych danych
* data_to_hdf5.py - konwersja danych z typu .dat na .h5
* tf_reader_example.ipynb - przykład wczytywania danych do Tensorflow (deprecated)
* scripts/Deep.ipynb - głowny plik z modelem
* scripts/ICM_template.py - przykład modelu gotowego na ICM
* TensorBoard.ipynb - notatnik do wyświetlania wyników z TensorBoard
* scripts/ready_to_ICM/ - folder z testowanymi modelami
* scripts/Plots-2.ipynb - tworzenie wykresów
* results/ - wyniki poszczególnych modeli

Środowisko Python - kontener Docker pkaleta57/custom-tf:2.15.0-gpu-jupyter

UPDATE: W celu przyspieszenia treningu po wczytaniu datasetu jest on zapisywany w formacie odpowiednim dla Tensorflow przez komendę dataset.save(YOUR_PATH) po wczytaniu tak jak w scripts/Deep.ipynb. Dalej używanie takie jak w scripts/ICM_template.py.

Testowane kierunki:

* różne funkcje straty - najlepsze MSE dzielone przez sqrt(E)
* 2 architektury modelu (w scripts/models) - większy model wypadał lepiej ale szybciej się przeucza
* regularyzacja CONV- najlepiej wypada regularyzacja L2 z wartością 1e-4
* regularyzacja Dense - najlepiej wypada regularyzacja L1L2 (L1=1e-3, L2=1e-2) lub L2 (1e-3)
* Batch Normalization - wydaje się że poprawia wyniki
* Dropout - Trzeba ostrożnie dobierać wartości. Testy wykazały, że najlepszy efekt dawało nałożenie dropoutu 0.2 na trzecią warstwę konwolocyjną. Narzucenie dropoutu rzędu 0.2 na warstwy gęste także delikatnie polepszyło zachowanie modelu. W kontekście warstw konwolucyjnycj, paradoksalnie 'klasyczny' dropout wydawał się działać lepiej niż tzw. 'spatial dropout'

Najlepszy model:
convFilters=[32,32,16], denseNeurons=[64,32,16], reg_conv=regularizers.l2(1e-4), reg_dense=regularizers.L1L2(l1=1e-3, l2=1e-2), LossFunction=snorm_MSE, Batch Normalization po każdej warstwie konwolucyjnej.

Dalsze propozycje:

* sprawdzenie lekko zmienionej architektury sieci
* dalsze badanie regularyzacji
* inne metody zwalczające przeuczanie
* metody niwelujące duże oscylacje funkcji straty na zbiorze walidacyjnym
