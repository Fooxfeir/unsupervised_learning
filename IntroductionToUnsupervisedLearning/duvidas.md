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

- **Página 10, aula 4** 
- **Description of Doubt:** 
    Como seriam os mapas de 3 dimensões do espaço latente de um auto-encoder convolucional?
    

## Context
- **Relevant Concepts:**  
    _List any related concepts or topics._

- **Attempts to Resolve:**  
    _Describe what you tried to understand or solve the doubt._

## Additional Notes
- **References:**  
    _Mention any external resources or textbook sections consulted._

- **Screenshots/Quotes (if needed):**  
    _Attach or quote relevant parts from the PDF._