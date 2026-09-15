# CKP01 — Chatbot Profissional · Tutor ENEM

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**

**Integrantes:** Davi Queiroz Zuolo (571669) · Gustavo Zagato (569420) · Daniel Vilela Mana (571632) · Kayo Henderson (570706)

## Domínio

**Tutor ENEM** — chatbot educacional para estudantes do ensino médio e candidatos ao ENEM.

O sistema possui uma **lista fechada de 5 matérias**. O aluno escolhe uma matéria e conversa com um tutor especializado naquele domínio. Cada matéria possui sua própria persona, escopo, regras e sala de memória.

| Matéria                 | Tutor(a) | Área do ENEM                  |
| ----------------------- | -------- | ----------------------------- |
| ✍️ Redação              | Clara    | Redação                       |
| 📐 Matemática           | Teo      | Matemática e suas Tecnologias |
| 🌎 História & Geografia | Helena   | Ciências Humanas              |
| 🧬 Biologia             | Bia      | Ciências da Natureza          |
| ⚡ Física                | Max      | Ciências da Natureza          |

### Por que este domínio?

O ENEM é um domínio educacional adequado para aplicação de técnicas de engenharia de prompts, memória conversacional e saída estruturada.

O chatbot foi projetado para auxiliar o estudante dentro da matéria escolhida, explicando conceitos, orientando o raciocínio e evitando respostas fora do escopo daquela sala.

O domínio também foi escolhido pensando na evolução do projeto ao longo do semestre, mantendo o **Tutor ENEM como base** para as próximas etapas previstas pelo curso.

### Usuários-alvo

* Estudantes do ensino médio;
* Pessoas que estão se preparando para o ENEM;
* Alunos que desejam revisar conteúdos e compreender o raciocínio por trás das questões.

Como parte dos usuários pode ser menor de idade, o sistema possui regras de segurança e linguagem adequada ao contexto educacional, além de restrições contra solicitação e exposição de dados pessoais.

---

## Requisitos atendidos

| Requisito            | Status | Implementação                                                                      |                               |
| -------------------- | ------ | ---------------------------------------------------------------------------------- | ----------------------------- |
| Pipeline LCEL        | ✅      | `chain.py` — composição com operador `                                             | ` para as chains estruturadas |
| ChatOllama           | ✅      | `gemma4:cloud` via Ollama                                                          |                               |
| ChatPromptTemplate   | ✅      | `prompts.py` — mensagens `system` e `human` separadas e variáveis via `.partial()` |                               |
| 2 chains             | ✅      | `ConversationChain` para conversa + pipeline LCEL para saídas estruturadas         |                               |
| Memória gerenciada   | ✅      | `ConversationChain` + `ConversationTokenBufferMemory`, limite de 1200 tokens       |                               |
| Pydantic v2          | ✅      | `schemas.py` — `CorrecaoRedacao` e `RelatorioSessao`                               |                               |
| PydanticOutputParser | ✅      | Integrado às chains de saída estruturada                                           |                               |
| System prompt        | ✅      | `prompts.py` — persona, escopo, regras, segurança e XML tagging                    |                               |
| Domínio documentado  | ✅      | Este README + `prompts.py`                                                         |                               |
| Context engineering  | ✅      | `context_rot.py` — experimentos A, B1 e B2 com contexto crescente                  |                               |
| Contagem de tokens   | ✅      | `tiktoken` + tokens reportados pelo Ollama                                         |                               |
| Métricas de contexto | ✅      | qualidade, tokens, latência e métricas individuais de recuperação                  |                               |
| Meta prompting       | ⏳      | Diferencial opcional não implementado                                              |                               |

Os requisitos de memória seguem a orientação da Aula 02 para tutor educacional/assistente de estudo, na qual `ConversationTokenBufferMemory` é indicada porque as explicações mais recentes tendem a ser mais relevantes, com limite sugerido de 1000–2000 tokens.

A saída estruturada utiliza `PydanticOutputParser`, conforme a orientação da Aula 03 para integração do Pydantic v2 às chains LCEL.

---

## Como executar

O projeto foi desenvolvido para execução **local, sem Google Colab**.

### 1. Criar o ambiente virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Criar o arquivo `.env`

Copie o arquivo de exemplo:

Windows:

```bash
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Preencha a variável `OLLAMA_API_KEY` conforme o ambiente utilizado.

> **Importante:** o arquivo `.env` contém credenciais e não deve ser enviado para o repositório ou para o `.zip` da entrega. Apenas `.env.example` deve ser entregue.

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o chatbot

```bash
python -m app.main
```

A interface Gradio será disponibilizada localmente em:

```text
http://localhost:7860
```

---

## Comandos de demonstração e evidências

### Memória conversacional

```bash
python -m app.memory_manager
```

Executa a demonstração da memória em múltiplos turnos e apresenta o estado da memória.

A Aula 02 exige a demonstração de múltiplos turnos e a inspeção do estado da memória; o projeto utiliza esse fluxo com `ConversationTokenBufferMemory`.

### Context Rot — experimento A

```bash
python -m app.context_rot
```

Executa o experimento de crescimento do contexto com distrações genéricas.

O resultado é salvo em:

```text
context_rot_resultados.md
```

### Context Rot — experimento B1

```bash
python -m app.context_rot --b1 --repeticoes 3
```

Executa o experimento de posição do fato relevante no contexto, mantendo o mesmo tamanho de contexto e alterando sua posição entre:

```text
inicio → meio → fim
```

### Context Rot — experimento B2

```bash
python -m app.context_rot --b2 --repeticoes 3
```

Executa o experimento com contexto crescente, informações semanticamente concorrentes e tarefa composta.

### Guardrails

```bash
python -m app.guardrails
```

Executa os casos de teste relacionados às proteções de entrada e saída.

---

## Arquitetura — 2 Chains

A arquitetura segue a estrutura ensinada na Aula 03: uma chain de conversa com memória e uma segunda chain baseada em LCEL para saída estruturada.

```text
                         ┌──────────────────────────────┐
                         │          Gradio              │
                         │          main.py             │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │         Guardrails            │
                         │       guardrails.py           │
                         └──────────────┬───────────────┘
                                        │
                                        ▼

           ┌────────────────────────────────────────────────┐
           │                Chain 1 — Chat                  │
           │                                                │
           │ ConversationChain                              │
           │   ├── ChatPromptTemplate                       │
           │   ├── System Prompt da matéria                 │
           │   ├── {history}                                │
           │   ├── {input}                                  │
           │   ├── ChatOllama — gemma4:cloud                │
           │   └── ConversationTokenBufferMemory — 1200    │
           │                                                │
           └──────────────────────┬─────────────────────────┘
                                  │
                                  ▼
                         ┌──────────────────────┐
                         │  Guardrail de saída  │
                         └──────────────────────┘


           ┌────────────────────────────────────────────────┐
           │            Chain 2 — Estruturada               │
           │                                                │
           │ ChatPromptTemplate                             │
           │          │                                     │
           │          ▼                                     │
           │ ChatOllama                                     │
           │          │                                     │
           │          ▼                                     │
           │ PydanticOutputParser                            │
           │                                                │
           │   ├── CorrecaoRedacao                           │
           │   └── RelatorioSessao                           │
           │                                                │
           └────────────────────────────────────────────────┘
```

A Chain 1 é responsável pela conversa do aluno e utiliza memória gerenciada.

A Chain 2 utiliza LCEL para produzir saídas estruturadas e validadas por schemas Pydantic.

A utilização de `PydanticOutputParser` em vez de `with_structured_output()` segue diretamente a orientação da Aula 03 para o CKP01.

---

## System Prompt e especialização por matéria

O system prompt possui uma estrutura comum a todas as matérias e campos específicos de cada domínio.

A estrutura utiliza marcação XML para separar as responsabilidades do prompt:

```text
<identidade>

<público>

<objetivo>

<escopo>

<fora_do_escopo>

<regras_gerais>

<regras_materia>

<seguranca>

<formato_resposta>

<exemplos>

<lembrete_final>
```

A parte específica da matéria define:

* Persona do tutor;
* Especialidade;
* Escopo permitido;
* Conteúdos fora do escopo;
* Regras específicas;
* Exemplos de comportamento.

Dessa forma, cada sala mantém sua especialização.

Por exemplo, uma pergunta de Biologia feita na sala de Matemática não é respondida como conteúdo de Biologia: o tutor orienta o aluno a utilizar a sala correspondente.

A utilização de XML tagging faz parte do conteúdo de Context Engineering apresentado na Aula 04.

---

## Estrutura do projeto

```text
app/

├── __init__.py
├── main.py              # Interface Gradio e ponto de entrada
├── chain.py             # Chains LCEL e orquestrador TutorENEM
├── memory_manager.py    # ConversationTokenBufferMemory por sala
├── schemas.py           # Schemas Pydantic v2
├── context_rot.py       # Experimentos de Context Rot
├── prompts.py           # System prompts e configurações das matérias
├── guardrails.py        # Validação e proteção de entrada/saída
└── tokens.py             # Contagem de tokens com tiktoken

.env.example             # Modelo das variáveis de ambiente
requirements.txt         # Dependências do projeto
README.md                # Documentação
```

---

## Justificativa da memória

### Escolha: `ConversationTokenBufferMemory`

O projeto utiliza:

```text
ConversationTokenBufferMemory

max_token_limit = 1200
```

A escolha segue a recomendação apresentada na Aula 02 para **tutores educacionais e assistentes de estudo**: a memória TokenBuffer mantém uma janela deslizante das mensagens mais recentes, sendo adequada quando as explicações recentes são mais relevantes para a próxima interação. A aula indica uma faixa de 1000–2000 tokens para esse cenário.

### Por que TokenBuffer?

Uma sessão de estudo pode possuir muitos turnos. Manter todo o histórico indefinidamente aumenta o contexto enviado ao modelo.

A `ConversationTokenBufferMemory` mantém as mensagens mais recentes até o limite configurado e descarta as mais antigas quando o limite é atingido. Esse comportamento corresponde à janela deslizante apresentada na Aula 02.

O limite escolhido foi de **1200 tokens**, dentro da faixa de **800–1500 tokens** definida para o CKP01 e também dentro da faixa recomendada na Aula 02 para tutor educacional.

### Por que não BufferMemory?

`ConversationBufferMemory` mantém todo o histórico da conversa. Em sessões longas, isso faz o número de tokens crescer continuamente.

### Por que não SummaryMemory?

`ConversationSummaryMemory` gera um resumo progressivo utilizando o LLM. Embora possa reduzir o tamanho do histórico, existe o risco de informações específicas serem resumidas ou omitidas. Além disso, a Aula 02 destaca o custo adicional de uma chamada ao LLM para atualização do resumo.

### Trade-off

A principal consequência da TokenBuffer é que informações muito antigas podem sair da janela.

Esse comportamento é intencional: o projeto prioriza **controle previsível do tamanho do contexto** e relevância das interações recentes.

A interface mantém a transcrição da sessão separadamente para permitir a geração do relatório completo da sessão.

---

## Memória por sala

A memória é isolada por:

```text
sessão + matéria
```

Isso impede que o histórico de uma matéria seja utilizado em outra.

Exemplo:

```text
Sessão A + Matemática

        ≠

Sessão A + Biologia
```

Assim, uma conversa realizada na sala de Física não é incorporada ao contexto da sala de Biologia.

A demonstração da memória pode ser executada com:

```bash
python -m app.memory_manager
```

O teste utiliza múltiplos turnos para demonstrar a persistência do contexto, conforme solicitado na Aula 02.

---

## Pydantic v2 e saída estruturada

O projeto utiliza **Pydantic v2** para validar saídas estruturadas.

Os principais schemas são:

### `CorrecaoRedacao`

Utilizado para estruturar a correção de uma redação.

### `RelatorioSessao`

Utilizado para estruturar o relatório da sessão de estudos.

As saídas são processadas por:

```text
ChatPromptTemplate
        |
ChatOllama
        |
PydanticOutputParser
```

O parser valida a resposta produzida pelo modelo de acordo com o schema definido.

A Aula 03 define o schema como o contrato entre os dados produzidos pelo LLM e o que o código espera receber, com campos tipados e validados pelo Pydantic.

---

## Guardrails

O projeto possui proteção em camadas para manter o chatbot dentro do escopo definido.

### 1. Validação de entrada

As entradas são verificadas antes da chamada ao modelo.

São considerados, entre outros:

* tamanho da mensagem;
* padrões de prompt injection;
* tentativa de revelar instruções internas;
* tentativa de alterar a identidade do tutor;
* tentativa de ativar modo irrestrito;
* tags falsas;
* blocos Base64;
* dados pessoais como CPF, e-mail e telefone.

Uma entrada bloqueada não chama o modelo e não é armazenada na memória.

### 2. Separação entre dados e instruções

A pergunta do aluno é encapsulada em uma tag específica:

```xml
<pergunta_usuario>
...
</pergunta_usuario>
```

Da mesma forma, uma redação é tratada separadamente:

```xml
<redacao>
...
</redacao>
```

### 3. Proteção no system prompt

O prompt possui regras específicas de segurança e um lembrete final para reforçar o comportamento esperado.

### 4. Validação da saída

A resposta do modelo é analisada antes de ser apresentada ao usuário.

Caso sejam detectadas informações que não deveriam aparecer, a resposta é substituída e a memória é corrigida para não manter a resposta inadequada.

### 5. Logging

Bloqueios realizados pelos guardrails são registrados no terminal para facilitar testes e auditoria.

---

## Context Rot

O projeto possui experimentos específicos para analisar o comportamento do modelo quando o contexto cresce.

O conceito de Context Engineering apresentado na Aula 04 trata o contexto como um recurso que envolve, entre outros elementos, system prompt, dados e histórico, além de abordar o fenômeno de **context rot**, no qual a qualidade pode degradar em contextos longos.

No projeto, os experimentos mantêm controlados o modelo, o system prompt e a pergunta final, enquanto variam o conteúdo e/ou o tamanho do contexto.

O fato principal utilizado nos experimentos é:

```text
Meu nome é Ana, estou no 3º ano e minha meta em Matemática no ENEM é 780 pontos.
```

A avaliação verifica se o modelo consegue recuperar corretamente informações do perfil da aluna.

### Experimento A — crescimento do contexto

O experimento A mantém um fato relevante e aumenta progressivamente a quantidade de distrações genéricas.

As janelas testadas foram:

```text
0 → 5 → 10 → 15 → 20 turnos de distração
```

Nesse experimento, **não foi observada degradação mensurável de qualidade no intervalo testado**.

Esse resultado é mantido como evidência experimental, sem alterar artificialmente o experimento para produzir uma queda.

### Experimento B1 — posição do fato

O B1 mantém a mesma quantidade de contexto e altera apenas a posição do fato relevante:

```text
início
meio
fim
```

Configuração executada:

```text
160 turnos de distração
3 posições
3 repetições por posição
```

Resultado observado:

| Posição | Turnos | Tokens tiktoken | Tokens Ollama | Qualidade |
| ------- | -----: | --------------: | ------------: | --------: |
| início  |    160 |           49004 |         40546 |      100% |
| meio    |    160 |           49004 |         40546 |      100% |
| fim     |    160 |           49004 |         40546 |      100% |

As três posições apresentaram **100% de qualidade nas três repetições**.

Portanto, no cenário testado, a simples posição do fato dentro do contexto não produziu degradação mensurável.

### Experimento B2 — competição semântica e tarefa composta

O B2 amplia o experimento para uma tarefa que exige a integração de múltiplas informações do perfil da aluna.

Além do nome e da meta, o modelo precisa identificar uma dificuldade e determinar qual assunto deve ser priorizado.

O contexto também contém perfis concorrentes com nomes, metas e dificuldades diferentes.

As janelas executadas foram:

```text
0 → 20 → 40 → 80 → 160 → 320 turnos de distração
```

Configuração:

```text
6 janelas
3 repetições por janela
mesmo modelo
mesmo system prompt
mesma pergunta final
```

Resultado obtido:

| Turnos | Tokens tiktoken | Tokens Ollama | Ana | Meta | Prioridade | Sem confusão | Qualidade |
| -----: | --------------: | ------------: | --: | ---: | ---------: | -----------: | --------: |
|      0 |            1837 |          1705 |  0% | 100% |       100% |         100% |       75% |
|     20 |            7774 |          6590 |  0% |   0% |         0% |           0% |        0% |
|     40 |           13860 |         11600 |  0% |   0% |         0% |          67% |       17% |
|     80 |           25883 |         21495 |  0% |   0% |         0% |         100% |       25% |
|    160 |           50078 |         41410 |  0% |   0% |         0% |          33% |        8% |
|    320 |           98319 |         81115 |  0% |   0% |         0% |         100% |       25% |

A métrica de qualidade do B2 é composta por quatro critérios binários:

1. Identificou Ana;
2. Lembrou a meta de 780 pontos;
3. Identificou Probabilidade como prioridade;
4. Não confundiu o perfil com os distratores.

Por utilizar quatro critérios binários, a qualidade percentual assume valores em incrementos de 25%.

O resultado mais relevante não é uma queda monotônica com o aumento dos tokens. O comportamento observado é **oscilante**, porém apresenta forte degradação em relação ao cenário inicial quando o contexto passa a conter informações semanticamente concorrentes.

Em algumas janelas, o modelo recuperou informações pertencentes a outro perfil, como **Lucas, 760 pontos e Estatística**, em vez das informações da aluna Ana.

Esse comportamento demonstra uma forma de confusão de atribuição dentro de um contexto semanticamente competitivo.

### Conclusão dos experimentos

Os três experimentos são complementares.

O **Experimento A** não apresentou degradação mensurável no intervalo testado.

O **Experimento B1** também não apresentou degradação quando apenas a posição do fato foi alterada.

Já o **Experimento B2**, ao introduzir informações semanticamente concorrentes e exigir a integração de múltiplos atributos do perfil, apresentou degradação de qualidade e confusão entre perfis conforme o contexto cresceu, embora de maneira não monotônica.

Portanto, os resultados não sustentam a afirmação de que **todo aumento de contexto necessariamente reduz a qualidade**. Eles sustentam que a **composição do contexto e a competição semântica podem ser fatores relevantes para a degradação do desempenho**, além do tamanho do contexto.

Os experimentos não foram ajustados artificialmente para produzir uma queda de qualidade.

---

## Diferencial: métricas de contexto

Além da avaliação de qualidade, os experimentos coletam:

* tokens medidos pelo `tiktoken`;
* tokens de entrada reportados pelo Ollama;
* latência da chamada;
* qualidade percentual;
* métricas individuais de recuperação do contexto.

Isso permite comparar não apenas a resposta final, mas também o crescimento do custo de contexto e o comportamento do modelo sob diferentes condições experimentais.

---

## Segurança e escopo

O Tutor ENEM possui **escopo fechado por matéria**.

O sistema deve:

* responder em português brasileiro;
* permanecer dentro da matéria selecionada;
* explicar o raciocínio e não apenas fornecer respostas;
* adaptar a explicação quando o aluno apresentar dificuldade;
* evitar inventar informações;
* não revelar instruções internas do sistema;
* não solicitar dados pessoais desnecessários;
* recusar conteúdos fora do propósito educacional definido.

As regras são aplicadas no system prompt e reforçadas pelos guardrails.

---

## Relação com o semestre

O **CKP01 é a base do projeto Tutor ENEM**.

O domínio e a estrutura foram definidos de forma modular para permitir a evolução prevista no curso, mantendo o mesmo domínio ao longo dos próximos checkpoints.

O escopo deste repositório, entretanto, permanece limitado aos requisitos do **CKP01**:

* LangChain;
* LCEL;
* ChatOllama;
* memória gerenciada;
* Pydantic v2;
* Context Engineering;
* documentação do domínio.

Funcionalidades previstas para etapas posteriores do semestre não fazem parte da implementação deste checkpoint.

A própria documentação do curso estabelece o CKP01 como base para os próximos checkpoints, incluindo RAG e agentes.

---

## Checklist de execução

Antes da entrega:

```text
[ ] Criar .env a partir de .env.example

[ ] Configurar OLLAMA_API_KEY

[ ] Instalar requirements.txt

[ ] Executar python -m app.main

[ ] Testar as 5 matérias

[ ] Executar demonstração da memória

[ ] Executar experimento de Context Rot A

[ ] Executar experimento de Context Rot B1

[ ] Executar experimento de Context Rot B2

[ ] Verificar os guardrails

[ ] Confirmar que .env não está no Git

[ ] Entregar somente .env.example
```

---

## Observações para entrega

O arquivo `.env` **não deve ser enviado** na entrega, pois contém credenciais.

O pacote de entrega deve conter o projeto e o arquivo `.env.example`, conforme especificado no enunciado do CKP01.

O projeto deve ser executado localmente com:

```bash
python -m app.main
```

O CKP01 utiliza exclusivamente o modelo definido para o checkpoint:

```text
gemma4:cloud
```

---

**Disciplina:** Prompt Engineering and Artificial Intelligence

**FIAP · Ciência da Computação · 2º Semestre 2026**
