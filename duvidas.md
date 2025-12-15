# Doubt Template for Class PDF

## Class Information
- **Class Name:** 
- **Date:** 
- **PDF Title/Section:** 

## Doubt/Question
- **Página 6, aula 1** 
- **Description of Doubt:**  
    Quanto aos modelos autoregressivos, não ficou clara qual seria a ideia dele. Uma descrição melhor do que significa x < i (todos os elementos antes de i? Apenas o anterior?) poderia ser adicionada.

    Todo esse slide poderia ser refatorado:

    1) Deixar cada uma dessas ideias um pouco mais claras, sem necessariamente entrar em  detalhes

    2) Retirar o slide para que ele se torne uma breve menção e não confunda os alunos

- **Página 10, aula 1** 
- **Description of Doubt:**  
    Adicionar quais são as condições que fazem com que o CLT possa ser usado mesmo em casos nos quais $x_i$ possuem distribuições diferentes

- **Página 13, aula 1** 
- **Description of Doubt:**  
    Para que a interpretação da entropia como a quantidade de perguntas de sim ou não necessárias para se responder uma pergunta, o log precisa estar na base 2

- **Página 14, aula 1** 
- **Description of Doubt:**  
    O que significa o p($x_2$) na fórmula da divergência KL? Em tese, isso não está significando a probabilidade de $x_2$ assumir um valor fixo? Não deveria haver algum tipo de integração com respeito a $x_2$?

    De fato, isso está errado. Na verdade, o que estamos querendo modelar é a probabilidade de que outra distribuição q(x) gere aquela mesma leitura. Então será integrado sobre $x_1$

- **Página 14, aula 1** 
- **Description of Doubt:**  
    Existe uma forma muito intuitiva de explicar a divergência KL. 

    Tenha em mente que calcular essa divergência pode ser interpretado como gerar os dados de P(x) e analisar o quão bem Q(x) modela aqueles dados. 
    
    Olhando para o termo log(p(x)/q(x)), podemos derivar -log(q(x)/p(x)). Olhando o termo dentro do log
    
    Portanto, estamos perguntando quanta informação (o formato -log(q(x)) remete à entropia) é necessária para saber a saída de Q dado que sabemos a saída de P. Uma nota importante é que estamos ponderando isso usando a probabilidade de p(x) ao invés de q(x), o que faz sentido, pois precisamos levar em consideração a probabilidade de P gerar aquela informação. 

    Se a quantidade de informação necessária for 0, as distribuições são iguais. 

- **Página 7, aula 2** 
- **Description of Doubt:** 
    Precisamos definir melhor quem é x e quem é $x_i$

- **Página 14, aula 2** 
- **Description of Doubt:** 
    Explicar o Kronecker's delta

- **Página 6, aula 3** 
- **Description of Doubt:** 
    Um dos $Z_i$  seria simplesmente a distribuição de uma gaussiana em uma dimensão só?


- **Página 6, aula 3** 
- **Description of Doubt:** 
    Explicar melhor quais seriam esses momentos mais altos e por que eles não são mantidos

    O que são diferentes modos?
    R: Uma distribuição representada por mais de uma Gaussiana

    Por que eles perdem a capacidade de representar dependências não lineares?
    R: Como a variância está sendo explicada apenas em termos de combinações lineares entre os autovetores da matriz de covariância, a projeção resultante assume forma de simplesmente relações lineares
    D: Mas como relações não lineares seriam capturadas pela matriz de covariância antes da redução de dimensionalidade?

- **Página 9, aula 3** 
- **Description of Doubt:** 
    A matriz de Gram duplamente centrada só pode ser utilizada no caso em que a distância sendo utilizada é a euclidiana (caso em que o MDS degenera para um PCA), pois a estimativa em torno dos autovalores de B só funciona quando temos a igualdade
    
    ```math
    \|x_i - x_j\|^2 = \langle x_i, x_i \rangle + \langle x_j, x_j \rangle - 2\langle x_i, x_j \rangle
    ```

    Uma vez que encontrar os autovetores com maior autovalor de B corresponde a encontrar as projeções com maior valor do produto interno entre $x_i$ e $x_j$ e portanto minimizar a distância

    No caso em que estamos usando a distância euclidiana, por voltarmos a um PCA as desvantagens voltam a ser as mesmas? 

    Para usar distâncias diferentes o MDS precisa usar métodos de otimização para conseguir encontrar os vetores que melhor diminuem a diferença entre as distâncias

- **Página 9, aula 3** 
- **Description of Doubt:** 
    Adicionar comentários sobre efeitos de usar uma distância não-euclidiana


- **Página 7, aula 4** 
- **Description of Doubt:** 
    Como foi feito o cálculo da explosão de parâmetros?

    R: Considera-se a medida da base imagenet: imagens 224x224 com 3 canais


- **Página 10, aula 4** 
- **Description of Doubt:** 
    Como seriam os mapas de 3 dimensões do espaço latente de um auto-encoder convolucional?

    São as duas dimensões da projeção para cada um dos canais. As projeções dos canais ficam "empilhadas" e isso forma o terceiro eixo

- **Página 5, aula 4** 
- **Description of Doubt:** 
    Mudar a dimensão latente

## Context
- **Relevant Concepts:**  
    BATCH NORMALIZATION COM CNN:

        Uma convolução não passa de um produto interno entre o vetor representado pela CNN e o vetor representado por aquela região da imagem.

        Pensando nisso, existem patches que exemplificam dados de um mesmo tipo de textura na imagem. O que nós queremos é encontrar vetores (CNN) que tenham um alto produto interno com um patch

        No entanto, dois patches diferentes podem estar na mesma direção a partir do centro, embora em pontos diferentes. Isso quer dizer que uma CNN com ativação alta em um também teria alta ativação no outro, pois ela se importa com a direção dos vetores.

        Nesse caso, apenas o bias poderia nos salvar, pois ele poderia deslocar a origem dos vetores e discriminar entre diferentes clusters que estão na mesma direção a partir da origem (se a nova origem do vetor da camada de convolução estiver entre os dois clusters, ele terá ativação positiva para um e negativa para o outro)

        Uma alternativa é fazer Batch Normalization, pois isso centralizará todos os clusters na origem. Assim, menos clusters estarão na mesma direção a partir da origem e teremos menos pressão em aprender o bias (o que é geralmente difícil)

- **Tarefas a mais:**  
    Pegar a base do COREL e tentar fazer inicialmente o treinamento padrão supervisionado com apenas 10% dos dados (quantidade de dados anotados)
    Fazer um autoencoder convolucional com 50% dos dados, desplugar o decoder e plugar a cabeça fully connected e depois treinar com 10% dos dados que estão anotados. 
    Observar a diferença de acurácia.

## Additional Notes
- **References:**  
    _Mention any external resources or textbook sections consulted._

- **Screenshots/Quotes (if needed):**  
    _Attach or quote relevant parts from the PDF._