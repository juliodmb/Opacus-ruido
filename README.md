# Privacidade Diferencial no Treinamento de Redes Neurais: 

## testando o Opacus e o Impacto do Noise Multiplier

##  I. Introduçao   
### A aplicação de boas práticas de segurança é essencial  ao treinar redes neurais com datasets sensíveis como informações hospitalares ou dados valiosos. No  entanto, a limitação de hardware e memória, pode dificultar a implementação de técnicas que adicionam ruído aos gradientes para aumentar a privacidade do treinamento.                      Uma solução eficiente é o Opacus Ghost Clipping, que reduz o uso de memória ao evitar a necessidade de instanciar gradientes individuais. Atualmente, essa técnica funciona apenas para camadas lineares, o que exige um planejamento cuidadoso da arquitetura da rede.  
### Para maximizar sua eficácia, é recomendável utilizar um número maior de camadas lineares no início da rede, pois elas atuam diretamente sobre os pesos, o bias e o hiperplano. Dessa forma, o ruído se propaga melhor para as camadas não lineares subsequentes. 
### No entanto, adicionar ruído a camadas não lineares ainda é um desafio matemático complexo, sem uma solução definitiva. O Opacus possui dois mecanismos principais: recorte de gradiente por amostra e adição de ruído que, juntos, garantem uma privacidade mais robusta no treinamento. 
 
## II. Metodologia  
### Nesse trabalho, realizei testes para entender até que ponto o Noise Multiplier (multiplicador de ruído) pode ser elevado sem comprometer significativamente a predição. Esse parâmetro é crucial, pois, quanto maior ele for maior a privacidade, mas também maior o impacto na precisão do modelo. 
### Utilizando um dataset sintético, observei o comportamento do Noise Multiplier e sua relação com a qualidade da predição. 
### Para a 	construção do 	dataset utilizei torch.manual_seed(42) 	que 	garante 	a reprodutibilidade. Foram geradas 100.000 amostras,Conforme demostrado no quadro abaixo, a variável alvo y foi definida como uma combinação de múltiplos padrões matemáticos, componente senoidal, termo quadrático, interação cruzada, termo exponencial,  termo cúbico, ruído gaussian.    

![Texto Alternativo da Imagem](Imagem1.jpg)





## III. Resultados          
### Os resultados mostraram que um Noise Multiplier de 1 ofereceu um bom equilíbrio entre privacidade e predição, mantendo um ε (épsilon) com qualidade aceitável. 
![Texto Alternativo da Imagem](Imagem2.jpg)


  
###  O ε (épsilon) mede o nível de privacidade diferencial sendo:

###  •	Baixo ε (ex: 0,1 - 1,0) → Mais privacidade  
###  •	Alto ε (ex: 2 - 10 ou mais) → Menos privacidade  
###       Já com um Noise Multiplier maior que 1, a medição da privacidade não foi tão eficiente em relação a performace, perdendo muito do desempenho onde no noise 10 fica bem abaixo . 
### Dessa forma, garantir um equilíbrio entre segurança e performance é fundamental para treinar modelos robustos, seguros e adaptáveis a diferentes restrições computacionais. 
### Arquitetura da rede neural  artificial:      
![Texto Alternativo da Imagem](Imagem3.jpg)
