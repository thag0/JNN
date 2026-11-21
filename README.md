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
<img width="482" height="564" alt="Image" src="https://github.com/user-attachments/assets/cdc9418f-774f-45e0-bbc0-634c9b47d110" />

# *Exemplo de uso*

<img width="1701" height="647" alt="Image" src="https://github.com/user-attachments/assets/78ba9bcb-8eae-4931-82d7-9c8a5b25c805" />

*Exemplo de treino com o modelo sequencial, usando o otimizador sgd (Stochastic Gradient Descent) e função de perda mse (Mean Squared Error) usando dataset Iris*

# **Interface com código nativo**
Para acelerar processos críticos que demandam muito tempo em java, implementei a aceleração usando código nativo em C, por enquanto apenas algumas funcionalidades mais necessárias de imediato possuem suporte, mas tenho interesse em expandir esse cenário de aceleração e quem sabe futuramente implementar cuda.

Para ativar o uso de JNI basta adicionar estre trecho no início do programa:
```
JNNnative.on();
```

Se tudo der certo, a biblioteca carregará as dependências e a partir desse ponto usará aceleração via código nativo, em trechos que tiverem suporte.

Depois compile com

```
javac -cp "jnn.jar" NomeDoPrograma.java
```

E rode com

```
java --enable-native-access=ALL-UNNAMED -cp "jnn.jar" NomeDoPrograma
```

Importante reforçar que o jni por enquanto suporta apenas windows_x64 e que usa paralelismo via OpenMP.

# **Callbacks de treino**

É possível adicionar callbacks ao final de cada época de trieno para poder aproveitar os dados em processamentos para fazer análises especiais.

<img width="1370" height="600" alt="callback-example" src="https://github.com/user-attachments/assets/97e7a5d4-2a1f-4c56-985e-62ae9b7956d5" />

*Utilizando callback para capturar acurácia do modelo durante o treino*

# **Scheduler de treino**

Para melhorar o refinamento de treino, é possível utilizar Scherdulers, tipicamente usados para evitar saltos muito grandes no meio-fim do treino. Eles alteram a taxa de aprendizado do otimizador usado com o objetivo de melhorar a convergência e estabilidade ao decorrer do treino.

<img width="420" height="366" alt="scheduler-example" src="https://github.com/user-attachments/assets/db76cc3d-aaf0-4abe-8e23-a93f0622ff2e" />

*Configurando um scheduler*

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