# Laboratorio-3-Procesamiento
## Introducción
En esta práctica de laboratorio se realizó la captura y el procesamiento de señales de voz con el propósito de analizar sus características espectrales. Para ello, se aplicó la Transformada de Fourier como herramienta fundamental de análisis en frecuencia, lo que permitió observar el comportamiento de las componentes espectrales de cada señal. Posteriormente, se extrajeron parámetros característicos de la voz como la frecuencia fundamental, frecuencia media, brillo, intensidad, jitter y shimmer, con el fin de cuantificar las diferencias acústicas entre ambos géneros. Finalmente, se compararon los resultados obtenidos entre las señales de voz de hombres y mujeres, permitiendo desarrollar conclusiones sobre el comportamiento espectral de la voz humana en función del género.
## Importación de librerias 
Para el desarrollo de esta práctica se instalaron las siguientes librerías:
            
```python
import wave 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files
from scipy.io import wavfile
          
```
La librería `wave` se utilizó para leer, escribir, ver duración, canales e inspeccionar los archivos de audio `.wav`. Asimismo, `scipy.io.wavfile` se usó para leer las grabaciones de audio y extraer la frecuencia de muestreo `sr` y los datos de la señal `y`. 

## PARTE A

<p align="center">
<img src="diagrama1.png" width="400">

En la parte inicial del laboratorio, se grabó con un micrófono de celular la misma frase corta "Lo que tienes, muchos lo pueden tener, pero lo que eres, nadie lo puede ser" en 6 personas distintas: 3 hombres y 3 mujeres. Para esto, se usó el micrófonos de un teléfono para que las características de muestreo fueran las mismas para cada audio. Seguidamente, se guardó cada archivo de voz en formato `.wav`. Luego se importaron las señales de voz en Google Colab, se graficaron en el dominio del tiempo y se calculó la Transformada de Fourier de cada señal para graficar su espectro de magnitudes frecuenciales. Finalmente se identificaron las siguientes características de cada señal: Frecuencia fundamental, frecuencia media, brillo e intensidad (energía), para en la parte final realizar su respectivo análisis. 
