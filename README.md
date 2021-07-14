# LSTM Univariada
Aplicando Long short-term memory (LSTM) para prever dentro da série de 10 anos dos dados da estação do INMET de Porto Alegre. Explicação mais detalhada pode ser encontrada no meu artigo públicado no Medium, https://vlsantos5938.medium.com/cap-1-redes-neurais-aplica%C3%A7%C3%A3o-de-lstm-aos-dados-de-esta%C3%A7%C3%B5es-meteorol%C3%B3gicas-do-inmet-7d66bc0d2f45

Requisitos;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
from tqdm.notebook import trange
import random
from sklearn.metrics import f1_score,mean_absolute_error
from tensorflow.keras.datasets.boston_housing import load_data
from sklearn.preprocessing import MinMaxScaler
import glob

Desempenho da rede treinada; 
![image](https://github.com/vlsantos-bit/LSTM-Neural/blob/master/real_prev.png)

Distribuição dos dados;

![image](https://github.com/vlsantos-bit/LSTM-Neural/blob/master/distr.png)
