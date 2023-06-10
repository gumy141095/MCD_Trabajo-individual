# Redes Neuronales
## 1. Definición y ejemplos
El término red neuronal se aplica a una familia de modelos relacionada de manera aproximada que se caracteriza por un gran espacio de parámetro y una estructura flexible y que proviene de los estudios sobre el funcionamiento del cerebro. Conforme fue creciendo la familia, se diseñó la mayoría de los nuevos modelos para aplicaciones no biológicas, aunque gran parte de la terminología asociada refleja su origen.

Las definiciones específicas de redes neuronales son tan variadas como los campos en que se utilizan. Aunque ninguna definición única cubre correctamente toda la familia de modelos, por ahora, tenga en cuenta la siguiente descripción:

Una red neuronal es un procesador distribuido en paralelo de forma masiva con una propensión natural a almacenar conocimiento experimental y convertirlo en disponible para su uso. Asemeja al cerebro en dos aspectos:

* El conocimiento se adquiere por la red mediante un proceso de aprendizaje.
* Las fuerzas de conexión interneuronal, conocidas como ponderaciones sinápticas, se utilizan para almacenar el conocimiento.

La definición anterior plantean exigencias mínimas sobre la estructura y los supuestos del modelo. Por tanto, una red neuronal puede aproximar una amplia gama de modelos estadísticos sin que tenga que hipotetizar de antemano determinadas relaciones entre las variables dependientes e independientes. En lugar de eso, la forma de las relaciones se determina durante el proceso de aprendizaje. Si una relación lineal entre las variables dependientes e independientes es adecuada, los resultados de la red neuronal deben aproximarse lo máximo posible a los del modelo de regresión lineal. Si una relación no lineal es más adecuada, la red neuronal se aproximará automáticamente a la estructura del modelo "correcta".

### **Recomendador de YouTube**
YouTube es la compañía más grande del mundo para compartir, crear y visualizar contenido audiovisual. Las recomendaciones de YouTube son responsables de ayudar a más de mil millones de usuarios a descubrir contenido personalizado. Uno de los mayores retos que tuvieron que afrontar a la hora de crear el algoritmo es la cantidad de datos que son subidos a YouTube por segundo. Por lo tanto, una de estas redes neuronales tiene que tener la capacidad de ser sensible (responsive) tanto al último contenido subido a la plataforma como a las interacciones del usuario con esta.

### **Dynamic pricing Amazon**
Amazon es el líder indiscutible del comercio electrónico. Es conocido por todos que utiliza precios dinámicos. Según un estudio, Amazon varía los precios más de 2.5 millones de veces al día. El reto de esta red neuronal es que los precios en la era digital deben fijarse en tiempo real basándose en la oferta y la demanda de un determinado producto durante un limitado periodo de tiempo. Compañías como Walmart o Uber utilizan estos algoritmos para ofrecer precios más competitivos a sus clientes.

### **Identificar riesgos en banca**
HSBC es una de los bancos que utiliza redes neuronales para transformar la forma de procesar los préstamos e hipotecas. Esta compañía usa este tipo de algoritmos de inteligencia artificial para analizar el comportamiento de antiguos clientes y así poder dar una estimación del riesgo para un cliente nuevo a la hora de adquirir una hipoteca o préstamo.

### **Personalizar las estrategias de marketing**
En los últimos años son varias las compañías que utilizan inteligencia artificial para mejorar sus estrategias de marketing. Las redes neuronales son algoritmos que pueden procesar gran cantidad de datos como: perfiles de compradores, patrones de compra u otros tipos de datos específicos para cada empresa. Este tipo de características hacen que sean los algoritmos perfectos para analizar el mercado y proponer una estrategia de marketing personalizada por cliente. Sephora o Starbucks son dos de las compañías que utilizan este tipo de inteligencia artificial para incrementar sus beneficios.

En general, algunos tipos de redes neuronales serian:

- Redes Neuronales Feedforward (FNN):

Las redes neuronales feedforward, también conocidas como redes neuronales de propagación hacia adelante, son la forma más básica y fundamental de redes neuronales. En estas redes, la información fluye en una dirección, desde la capa de entrada hacia la capa de salida, sin ciclos ni bucles. Cada neurona en una capa está conectada a todas las neuronas de la capa siguiente. Son ampliamente utilizadas para problemas de clasificación y regresión.

- Redes Neuronales Convolucionales (CNN):

Las redes neuronales convolucionales están diseñadas específicamente para el procesamiento de datos en forma de cuadrículas, como imágenes. Estas redes utilizan capas de convolución para extraer características espaciales de las imágenes y capas de agrupación para reducir la dimensionalidad. Las CNN han demostrado un gran éxito en la visión por computadora, especialmente en tareas de reconocimiento de objetos y clasificación de imágenes.

- Redes Neuronales Recurrentes (RNN):

Las redes neuronales recurrentes son adecuadas para modelar datos secuenciales, donde la información pasa de una neurona a otra en bucles o retroalimentación. Cada neurona en una capa recurre a sí misma o a las neuronas anteriores en la secuencia. Estas redes son capaces de capturar la dependencia temporal en los datos y son ampliamente utilizadas en tareas de procesamiento del lenguaje natural, traducción automática, reconocimiento de voz y generación de texto.

- Redes Neuronales de Memoria a Corto y Largo Plazo (LSTM):

Las redes neuronales LSTM son una variante de las redes neuronales recurrentes diseñadas para superar los problemas de desvanecimiento y explosión del gradiente que pueden ocurrir en las RNN estándar. Las LSTM utilizan unidades de memoria especializadas llamadas "celdas" que pueden almacenar información a largo plazo. Esto permite a las LSTM aprender dependencias a largo plazo y manejar secuencias más largas de manera más efectiva.

- Redes Neuronales Generativas Adversariales (GAN):

Las redes neuronales generativas adversariales son un tipo especial de red neuronal que consta de dos partes: un generador y un discriminador. El generador crea nuevas muestras de datos que se parecen a las del conjunto de entrenamiento, mientras que el discriminador intenta distinguir entre las muestras generadas y las reales. Estas redes se utilizan para generar contenido sintético, como imágenes y texto, y han demostrado un gran éxito en tareas de generación de imágenes realistas.
## 2. Esquema general
En general, una red neuronal esta conformada por una capa de entrada, una capa de salida y usualmente capas ocultas, dichas capas tienen conexiones neuronales entre ellas que a su vez tienen asignado un peso (el cual variara a lo largo del entrenamiento de la red) y por lo general, una funcion de activacion que modifica los valores obtenidos dependiendo de la funcion antes de propagarlos por la red neuronal. 

![red neuronal](https://www.futurespace.es/wp-content/uploads/2021/05/Motivacion-del-algoritmo-de-Backpropagation.jpg)

Las señales de entrada, la información que recibe nuestra red neuronal, son variables independientes. Los n-valores de entrada son multiplicados por sus respectivos pesos, es decir en la sinopsis el vector entrada  es multiplicado por el vector peso , dando como resultado una combinación lineal de las entradas y los pesos, algo que denominamos función de ponderación.

```math
X*W^t=(x_1,x_2,...,x_n)*\begin{pmatrix}w_1\\w_2\\ \vdots \\w_n\end{pmatrix} = \sum_{i=1}x_i*w_i 
```

con:

$x_i=$ valores de entrada

$w_i=$ peso de conexion neuronal

Por otro lado, el "sesgo" es un valor constante añadido que define la facilidad con la que se dispara una neurona y este se añade despues de considerar la sumatoria anterior.

Y por último dicho resultado se propaga a la salida (pasando en muchos casos antes por una funcion de activacion). Dicho valor puede ser la nueva entrada de una neurona, formando así las redes neuronales, o bien puede ser el resultado final, nuestra variable respuesta. Las respuestas obtenidas pueden ser una variable continua como el precio de un objeto, una respuesta binaria (0,1) (Sí, No) si una persona padece algún tipo de enfermedad o no, o puede ser una respuesta categórica que nos sirva para clasificar por ejemplo.

![diagrama de percepcion neuronal](https://empresas.blogthinkbig.com/wp-content/uploads/2019/11/Screenshot-2019-11-25-at-14.35.10.png?resize=1053%2C676)

### Función de activación

Una función de activación es una función que transmite la información generada por la combinación lineal de los pesos y las entradas, es decir son la manera de transmitir la información por las conexiones de salida. La información puede transmitirse sin modificaciones, estaríamos hablando de una función identidad, o bien que no transmita la información. Como lo que queremos es que la red sea capaz de resolver problemas cada vez más complejos, las funciones de activación generalmente harán que los modelos sean no lineales. Entre las funciones de activación más conocidas o más usadas se encuentran:

* Función Escalón, (similar a la función binaria.)
$$ \phi(x)=
\begin{cases}
  0\text{ si }x<0\\    
  1\text{ si }x\ge0 
\end{cases} $$
* Función Sigmoidal.
$$ \phi(x)=\frac{1}{1+e^{-x}}$$
* Función Rectificadora (ReLU).
$$ \phi(x)= max\{0,x\},\text{ siendo } x \ge 0$$
* Función Tangente Hiperbólica.
$$ \phi(x)= \frac{1-e^{-2x}}{1+e^{-2x}}$$

* Funciones de Base Radial. (Gausianas, multicuadráticas, multicuadráticas inversas…)

## 3. Problema elegido

El problema que he elegido para desarrollar es *detección de digitos escritos a mano*, que consiste en hacer uso de imagenes en blanco y negro de 28x28 de digitos escritos a mano para hacer que la red entrenada reconozca nuevas muestras que no se encuentren en la base de datos usada para entrenar (en este caso, el dataset mnist que ofrece tensorflow).

El reconocimiento de caracteres escritos a mano es una tarea desafiante pero fundamental en el campo de la visión por computadora y el procesamiento de imágenes. Es una habilidad cognitiva que los humanos realizan fácilmente, pero que ha sido un desafío para las máquinas durante mucho tiempo. En los últimos años, las redes neuronales convolucionales (CNN) han demostrado ser una poderosa herramienta para abordar esta tarea y han logrado avances significativos en el reconocimiento de caracteres escritos a mano.

## 4. Desarrollo matemático del problema elegido
En general, las imagenes utilizadas tienen dimension 28x28, por lo cual lo podemos interpretar por una matriz con 784 elementos con valores entre 0 y 1 (pues dichos valores originalmente entre 0-255 se normalizaran), la red neuronal utilizada sera una convolucional y constara con 2 capas convolucionales y 2 de agrupacion alteranadas entre si, una capa oculta densa y la capa de clasificacion de salida. Utilizaremos la funcion de activacion ReLu en cada capa, salvo en la ultima donde se aplicara softmax.

La capa convolucional es la primera capa colocada encima de la imagen de entrada. Se utiliza para extraer las características de una imagen. Las n × n neuronas de entrada de la capa de entrada se convolucionan con un filtro de m × m y, a cambio, entregan una salida de tamaño (n − m + 1) × (n − m + 1). Introduce no linealidad a través de una función de activación neuronal. Los principales contribuyentes de la capa convolucional son el campo receptivo, el paso, la dilatación y el relleno.

![capa convolucional](https://www.mdpi.com/sensors/sensors-20-03344/article_deploy/html/images/sensors-20-03344-g003-550.jpg)

La computación de las CNN está inspirada en la corteza visual de los animales. La corteza visual es una parte del cerebro que procesa la información transmitida desde la retina. Procesa información visual y es sensible a pequeñas subregiones de la entrada. De manera similar, se calcula un campo receptivo en una CNN, que es una pequeña región de una imagen de entrada que puede afectar a una región específica de la red. También es uno de los parámetros de diseño importantes de la arquitectura de la CNN y ayuda a establecer otros parámetros de la CNN. Tiene el mismo tamaño que el núcleo y funciona de manera similar a la visión foveal del ojo humano para producir una visión central nítida. El campo receptivo se ve influenciado por el paso, el agrupamiento, el tamaño del núcleo y la profundidad de la CNN. Campo receptivo (r), campo receptivo efectivo (ERF) y campo proyectivo (PF) son términos utilizados para calcular subregiones efectivas en una red. El área de la imagen original que influye en la activación de una neurona se describe utilizando el ERF, mientras que el PF es un recuento de neuronas a las que las neuronas proyectan sus salidas. El paso es otro parámetro utilizado en la arquitectura de la CNN. Se define como el tamaño del paso con el que se desplaza el filtro cada vez. Un valor de paso de 1 indica que el filtro se desliza píxel a píxel. Un tamaño de paso más grande muestra menos superposición entre las celdas. 



## Referencias
- Wikipedia contributors. (2023). Red neuronal artificial. In Wikipedia, The Free Encyclopedia. Recuperado el 09/06/2023 desde https://es.wikipedia.org/wiki/Red_neuronal_artificial
- Ahlawat S, Choudhary A, Nayyar A, Singh S, Yoon B. Improved Handwritten Digit Recognition Using Convolutional Neural Networks (CNN). Sensors. 2020; 20(12):3344. https://doi.org/10.3390/s20123344
- TensorFlow. (2023). MNIST dataset. Recuperado de: [Dataset de Tensorflow](https://www.tensorflow.org/datasets/catalog/mnist)
