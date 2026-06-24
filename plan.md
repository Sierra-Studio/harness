Quero que vc crie um plano para um harness com:
- Memory
- Skills
- MCP
- Tools
- Loop
- Como Provider vamos usar o OpenRouter.
- Como observabilidade quero que todos os steps sejam logados e o uso de tokens para entrada e saída estejam gravados no turn.

## Memory
Com relação a memory iremos adotar as seguintes estratégias.
Sempre iremos saber o tamanho máximo da janela de contexto do modelo usado. Desse total iremos descontar o número de tokens usados no system prompt. No número de tokens que sobrar, iremos destinar ele a memory. Consideramos memory as mensagens trocadas bem como os summary das mensagens anteriores.
O summary deve ser criado sempre que o limite de tokens for atingido. Nesse momento é feito uma cópia das 10% ultimas mensagens e então é criado um summary de todo o contexto do memory, excluindo o system prompt.

Além disso iremos adotar uma estratégia de check points onde a cada 20 turns iremos classificar o assunto tratado anteriormente com poucas palavras.

Toda essa estratégia de memória e mensagens deve ser gravada em um banco de dados postgress

## Tools

Iremos adotar duas estratégias juntas, as built-in tools e as Index Tools.
As built-in tools são as tools basicas GetTools, SearchTools, GetSkills e Bash. As Index Tools são aquelas que podem ser adicionadas via MCP ao sistema, imagine que elas podem ser muitas e adiciona-las todas de uma vez no system prompt seria um problema terrível pois não sobraria espaco para a memory.

## Observabilidade

Iremos logar cada passo da execução e devemos ter uma preocupaçao grande com a quantidade de tokens gastos. Os tokens devem ser registrados nos turns.

## Skills

Skills devem ser definidas associadas a um usuário no banco de dados. 
O harness deve ter uma habilidade especial de criar skills automaticamente. A cada 10 sessões de um usuário, fazemos uma chamada para identificar se há padrões nos pedidos que poderia se transformar em uma nova skill. Caso encontre, uma nova skill é adicionada ao usuário.

## Loop

O loop deve ter mecanismos para controle de limite de tokens gastos.

## Provider
Vamos utilizar o https://openrouter.ai/