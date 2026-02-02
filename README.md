# **Java Neural Networks Library**

**Biblioteca para manipulação de modelos de Redes Neurais**

JNN é uma pequena biblioteca feita inteiramente em java para criação e treinamento de modelos de Redes Neurais Artificiais. Possuindo o próprio ecossistema integrado para manipulação de estruturas baseadas em Tensores.

# **Tensor**

Um Tensor é uma estrutura de dados que atua como um array multidimensional, sendo a parte mais importante da biblioteca para manipulações de camadas e modelos mais complexos.

O Tensor pode ser criado a partir de arrays primitivos oferecidos pelo prórpio java, ou a partir da quantidade de dimensões desejadas.

# *Exemplo de criação*
```
double[][] arr = {
    {1, 2},
    {3, 4}
};

Tensor t = new Tensor(arr);
t.print();
```

# *Saída*
```
Tensor (2, 2) = [
    [[1.0, 2.0],
     [3.0, 4.0]]
]
```

# **Modelos**

A biblioteca atualmente possui duas APIs para criação de modelos, sendo eles: RedeNeural e Sequencial.

# *RedeNeural*
O modelo RedeNeural foi a base da construção de todo esse ambiente, nele que comecei a incorporar os moldes para a biblioteca.
- Ele é uma implementação focada em Multilayer Perceptrons (MLPs), onde conta com apenas camadas densas na sua estrutura, o que acaba sendo um pouco limitado dependendo do tipo de problema, pois não traz flexibilidade nas configurações específicas de suas camadas;
- Contudo o modelo tem um ótimo desemepenho nas suas tarefas esperadas, como classificação e regressão, podendo ser uma escolha mais leve e simples dependendo da necessidade.

# *Sequencial*

O modelo Sequencial foi pensado para criar estruturas mais complexas que envolvam camadas mais diversificadas, podendo empilhas uma lista de camadas em sua estrutura.

# *Exemplo de criação*
```
Sequencial modelo = new Sequencial(
    new Entrada(1, 28, 28),
    new Conv2D(24, new int[]{3, 3}, "relu"),
    new MaxPool2D(new int[]{2, 2}),
    new Conv2D(30, new int[]{3, 3}, "relu"),
    new MaxPool2D(new int[]{2, 2}),
    new Flatten(),
    new Densa(100, "tanh"),
    new Densa(NUM_DIGITOS_TREINO, "softmax")
);
```

<img width="1097" height="892" alt="Image" src="https://github.com/user-attachments/assets/c0bd172b-6ec9-456a-ac7c-2dbe3230d1d9" />

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

# **Adicionais**
Caso queira ajustar a quantidade de threads alocadas para a biblioteca, chame antes do inicio das inicializações:
```
int threads = //valor desejado
PoolFactory.setThreads(threads);
```