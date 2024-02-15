# Biblioteca de Redes Neurais

Estou adaptando meu modelo de MLP para o formato matricial, com o objetivo de tornar o modelo mais modular para aplicações futuras.

Nesse modelo o centro de tudo é a classe Tensor4D que realiza as operações mais importantes dentro da rede, com isso quero tornar o desempenho
da rede melhor, tentando paralelizar o uso das operações que são usadas a todo momento durante a execução do programa.

Estou conseguindo expandir a modularização da rede, criando uma camada base que é herdada para as novas camadas criadas, ela possui a base das 
implementações necessárias pro funcionamento de uma camada dentro dos modelos.

Aos poucos estou adicionando novos tipos de camadas, além de novas funcionalidades para as camadas já existentes.

# Implementação do modelo Sequencial

Estou criando uma api para um modelo sequencial de camadas, ele é uma generalização de um modelo base, assim como na api "RedeNeural", ambos possuem as mesmas implementações de métodos, mas o modelo sequencial tem sua especialidade de trabalhar com diferentes tipos de camadas.

Seguindo a ideia de generalização, os otimizadores também sofreram algumas mudanças para poder lidar com multiplos tipos de camadas. Agora lidam com formas vetorizadas dos kernels e bias das camadas, sem se preocupar com a formatação de cada uma em especifico.

![treino sequencial](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/368c7994-ccc9-4baa-8417-5d67c7e5320c)

*Exemplo de treino com o modelo sequencial, usando o otimizador sgd (Stochastic Gradient Descent) e função de perda mse (Mean Squared Error)*

Até o momento o modelo sequencial está lidando perfeitamente bem com os métodos já criados na primeira api "RedeNeural" as generalizações estão funcionando bem. 

# Treinamento de modelos Convolucionais

Estou testando os modelos convolucionais no conjunto de dados do MNIST, atualmente estou usando todos os 10 dítigos para treino, onde cada dígito possui 200 amostras.

Esse é o modelo que até agora teve a maior acurárcia entre os testes (91%)

![conv-mnist-91](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/841acff8-0fed-480e-8b04-773a35272f70)

Já estou bem satisfeito de ter conseguido esses resultados com o modelo convolucional, mas sempre há espaço para melhorias e quero continuar trabalhando nisso.

# Regularização

Iniciei um pouco os estudos sobre a regularização e como evitar que os modelos entrem no problema de sobreajuste (overfitting). Inicialmente parti da ideia de que seria mais fácil apenas criar modelos menores (em número de parâmetros e camadas) mas depois pesquisei um pouco mais sobre os métodos de regularização.

Por enquanto a implementação mais fácil de fazer foi criar a camada de Dropout, que aleatoriamente durante o treinamento desativa algumas unidades para que não tenham contribuição no resultado final.

A camada de Dropout é bem simples de usar, basta empilhar entre camadas (também funcionando com camadas que não sejam densas, porém não testado), como sugere na imagem de exemplo:

![exemplo dropout](https://github.com/thag0/Biblioteca-de-Redes-Neurais/assets/91092364/079fae89-dc01-4ea1-9b62-ef4dc3086200)

*Exemplo de modelo usando camadas de abandono (dropout), valores arbitrários.*