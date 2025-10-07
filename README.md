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
La librería `wave` se utilizó para inspeccionar, leer, escribir y ver la duración y los canales de los archivos de audio `.wav`, al igual que `scipy.io.wavfile` que se usó para también leer las grabaciones de audio y además extraer la frecuencia de muestreo `sr` y los datos de la señal `y`. La librería `Numpy` fue escencial para trabajar con los arreglos de amplitudes y tiempos, detectar los cruces por cero (identificar periodos de vibración), y para aplicar las fórmulas del Jitter y Shimmer, sumando, promediando y restando elementos. `matplotlib.pyplot` y `pandas` sirvieron para la visualización de los resultados, tanto en gráficas como en tablas respectivamente. Y fiinalmente, `a`

## PARTE A

<p align="center">
<img src="diagrama1.png" width="400">

En la parte inicial del laboratorio, se grabó con un micrófono de celular la misma frase corta "Lo que tienes, muchos lo pueden tener, pero lo que eres, nadie lo puede ser" en 6 personas distintas: 3 hombres y 3 mujeres. Para esto, se usó el micrófono de un teléfono para que las características de muestreo fueran las mismas para cada audio. Seguidamente, se guardó cada archivo de voz en formato `.wav`, se importaron estas señales en Google Colab y se graficaron en el dominio del tiempo de la siguiente manera: 

```python
# Lectura de cada archivo
sr1, y1 = wavfile.read('/content/drive/MyDrive/Hombre-1.wav')
sr2, y2 = wavfile.read('/content/drive/MyDrive/Hombre-2.wav')
sr3, y3 = wavfile.read('/content/drive/MyDrive/Hombre-3.wav')
sr4, y4 = wavfile.read('/content/drive/MyDrive/mujer1.wav')
sr5, y5 = wavfile.read('/content/drive/MyDrive/mujer-2.wav')
sr6, y6 = wavfile.read('/content/drive/MyDrive/mujer3.wav')

# Vectores de tiempo
t1 = np.linspace(0, len(y1)/sr1, num=len(y1))
t2 = np.linspace(0, len(y2)/sr2, num=len(y2))
t3 = np.linspace(0, len(y3)/sr3, num=len(y3))
t4 = np.linspace(0, len(y4)/sr4, num=len(y4))
t5 = np.linspace(0, len(y5)/sr5, num=len(y5))
t6 = np.linspace(0, len(y6)/sr6, num=len(y6))

# Gráficas
plt.figure(figsize=(10, 18))

plt.subplot(6, 1, 1)
plt.plot(t1, y1, color='royalblue')
plt.title("Hombre 1 - Señal de voz")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(6, 1, 2)
plt.plot(t2, y2, color='royalblue')
plt.title("Hombre 2 - Señal de voz")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(6, 1, 3)
plt.plot(t3, y3, color='royalblue')
plt.title("Hombre 3 - Señal de voz")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(6, 1, 4)
plt.plot(t4, y4, color='darkorange')
plt.title("Mujer 1 - Señal de voz")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(6, 1, 5)
plt.plot(t5, y5, color='darkorange')
plt.title("Mujer 2 - Señal de voz")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(6, 1, 6)
plt.plot(t6, y6, color='darkorange')
plt.title("Mujer 3 - Señal de voz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplots_adjust(hspace=0.6)
plt.suptitle("Señales de voz en el dominio del tiempo", fontsize=16, y=0.95)
plt.show()         
```
Y mostrando los siguientes resultados: 

<p align="center">
<img src="señales-de-voz-en-el-dominio-del-tiempo.png" width="400">




 y se calculó la Transformada de Fourier de cada señal para graficar su espectro de magnitudes frecuenciales. Finalmente se identificaron las siguientes características de cada señal: Frecuencia fundamental, frecuencia media, brillo e intensidad (energía), para en la parte final realizar su respectivo análisis. 


