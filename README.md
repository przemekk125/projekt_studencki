
# projekt_studencki

Projekt studencki - fizyka. Celem projektu jest analiza danych z symulacji kalorymetru za pomocą metod uczenia maszynowego (3D CNN).

* dane_info.txt - informacjw o surowych danych
* data_to_hdf5.py - konwersja danych z typu .dat na .h5
* tf_reader_example.ipynb - przykład wczytywania danych do Tensorflow
* Deep.ipynb - głowny plik z modelem
* TensorBoard.ipynb - notatnik do wyświetlania wyników z TensorBoard

Środowisko Python - kontener Docker pkaleta57/custom-tf:2.15.0-gpu-jupyter
