# **Java Neural Networks Library**

**Biblioteca para manipulação de modelos de Redes Neurais**

JNN é uma pequena biblioteca feita inteiramente em java para criação e treinamento de modelos de Redes Neurais Artificiais. Possuindo o próprio ecossistema integrado para manipulação de estruturas baseadas em Tensores.

# **Tensor**

Um Tensor é uma estrutura de dados que atua como um array multidimensional, sendo a parte mais importante da biblioteca para manipulações de camadas e modelos mais complexos.

O Tensor pode ser criado a partir de arrays primitivos oferecidos pelo prórpio java, ou a partir da quantidade de dimensões desejadas.

# *Exemplo de criação*
```
float[][] arr = {
    {1, 2},
    {3, 4}
};

Tensor t = new Tensor(arr);
t.print();
```

# *No console*
```
Tensor ([[1.0, 2.0],
         [3.0, 4.0]])
```

*Internamente todos os tensores trabalham com dados do tipo float32.*

# **Modelos**

A biblioteca conta uma API para criação de modelos sequenciais de camadas.

# *Exemplo de criação*
```
Sequencial modelo = new Sequencial(
    new Entrada(1, 28, 28),
    new Conv2D(24, new int[]{3, 3}),
    new ReLU(),
    new MaxPool2D(new int[]{2, 2}),
    new Conv2D(30, new int[]{3, 3}),
    new ReLU(),
    new MaxPool2D(new int[]{2, 2}),
    new Flatten(),
    new Densa(100),
    new Tanh(),
    new Densa(10),
    new Softmax()
);
```

<img width="1701" height="647" alt="Image" src="https://github.com/user-attachments/assets/78ba9bcb-8eae-4931-82d7-9c8a5b25c805" />

*Exemplo de treino com o modelo sequencial, usando o otimizador sgd (Stochastic Gradient Descent) e função de perda mse (Mean Squared Error) usando dataset Iris*

# **Interface com código nativo**
Para acelerar processos críticos que demandam muito tempo em java, implementei a aceleração usando código nativo em C, por enquanto essas funcionalidades estão beneficiando apenas as camadas Densa e Conv2D (que são as mais pesadas), mas tenho interesse em expandir esse cenário de aceleração e quem sabe futuramente implementar cuda.

Para ativar o uso de JNI basta adicionar estre trecho no início do programa:
```
JNNNative.jni = true;
```

Compile com

```
javac -cp "jnn.jar" NomeDoPrograma.java
```

E rode com

```
java --enable-native-access=ALL-UNNAMED -cp "jnn.jar" NomeDoPrograma
```

Importante reforçar que o jni por enquanto suporta apenas windows_x64 e que o usuário deve ter instalado o OpenMP na máquina.

# **Callbacks de treino**

É possível adicionar callbacks ao final de cada época de trieno para poder aproveitar os dados em processamentos para fazer análises especiais.

<img width="1406" height="600" alt="Image" src="https://github.com/user-attachments/assets/886fcf8a-acf4-4bda-86f1-02f1fb4f7e71" />

*Utilizando callback para capturar acurácia do modelo durante o treino*

# **Data Augmentation**

Iniciando um pouco para a área de visão computacional, adicionei suporte para uso de transoformações de dados, que normalmente são aplicadas no momento em que o dado é requerido no treino.

Esse tipo de aborgadem aumenta a capacidade de generalização de modelos focados na áreas, aumentando artificialmente o conjunto de dados.

<img width="838" height="451" alt="Image" src="https://github.com/user-attachments/assets/efd9761c-4848-4ee2-863c-374af08f9581" />

*Técnica de flip horizontal (HFlip)*

<img width="840" height="447" alt="Image" src="https://github.com/user-attachments/assets/f3b4ce60-b996-424b-acfc-df57c95dc696" />

*Técnica de corte aleatório (RandomCrop)*

# **Adicionais**
Caso queira ajustar a quantidade de threads alocadas para a biblioteca, chame antes do inicio das inicializações:
```
int threads = //valor desejado
PoolFactory.setThreads(threads);
```