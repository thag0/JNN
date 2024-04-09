# **Biblioteca para manipulação de modelos de Redes Neurais**

Criação de uma pequena biblioteca para manipulação de modelos de machine learning com foco em redes neurais artificiais.

# **Modelo RedeNeural**

O modelo RedeNeural foi a base da construção de todo esse ambiente, nele que comecei a incorporar os moldes para a biblioteca.

Ele é uma implementação focada em Multilayer Perceptrons, onde conta com apenas camadas densas na sua estrutura, o que acaba sendo um pouco limitado dependendo do tipo de problema, pois não traz flexibilidade nas configurações específicas de suas camdas.

Contudo o modelo tem um ótimo desemepenho nas suas tarefas esperadas, como classificação e regressão, podendo ser uma escolha mais leve e simples dependendo da necessidade.

# **Modelo Sequencial**

O modelo Sequencial foi criado pensando em uma limitação contida no modelo anterior.

Nele criamos um modelo muito mais modular e personalizável, podendo empilhar quantas camadas forem necessárias. Com isso aumento o grau de liberdade na criação de modelos e expando a disponibilidade de camadas que agora não estão mais restritas apenas à Densa.

![treino sequencial](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/368c7994-ccc9-4baa-8417-5d67c7e5320c)

*Exemplo de treino com o modelo sequencial, usando o otimizador sgd (Stochastic Gradient Descent) e função de perda mse (Mean Squared Error)*

# **Treinamento de modelos Convolucionais**

Indo além nas problemáticas que podem ser resolvida usando os modelos, fazendo uso do Sequencial, comecei a testar usando o dataset MNIST, que é considerado o "Hello World do Machine Learning".

Com a adição de uma nova problemática, surgiu a adição de novos métodos de resolução, com isso desenvolvi novas arquiteturas de camadas para treinar modelos para esse problema. 

Segue um exemplo:

![arq conv](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/260f7848-e564-477b-b296-9c657addff7a)

*Modelo convolucional criado para classificar os dígitos do dataset MNIST*

# **Regularização**

Iniciei um pouco os estudos sobre a regularização e como evitar que os modelos entrem no problema de sobreajuste (Overfitting). Inicialmente parti da ideia de que seria mais fácil apenas criar modelos menores (em número de parâmetros e camadas) mas depois pesquisei um pouco mais sobre os métodos de regularização.

Por enquanto a implementação mais fácil de fazer foi criar a camada de Dropout, que aleatoriamente durante o treinamento desliga algumas unidades para que elas não tenham contribuição no resultado final.

A camada de Dropout é bem simples de usar, basta empilhar entre camadas, como sugere na imagem de exemplo:

![exemplo-dropout](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/c18f5981-1fe5-49ef-b126-85eebc756e70)

*Exemplo de modelo usando camadas de abandono (dropout), valores arbitrários.*

# **Dificuldades**

Excluindo a parte de pesquisa e estudo as maiores dificuldades encontradas até o momento foram:

- **Performance**: Melhorias em nível de código para melhorar a execução dos algoritmos;

- **Multithread**: Melhorias em nível de hardware para tirar proveito de múltiplos processadores do computador. Atualmente tudo roda em single thread;

- **Tensor4D**: Melhorias de performance e maior generalização da classe de tensor que uso dentro do ambiente;