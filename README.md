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
| ⚡ Física               | Max      | Ciências da Natureza          |

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

| Requisito            | Status | Implementação                                                                      |
| -------------------- | ------ | ---------------------------------------------------------------------------------- |
| Pipeline LCEL        | ✅      | `chain.py` — composição com operador `\|` para as chains estruturadas              |
| ChatOllama           | ✅      | `gemma4:cloud` via Ollama                                                          |
| ChatPromptTemplate   | ✅      | `prompts.py` — mensagens `system` e `human` separadas e variáveis via `.partial()` |
| 2 chains             | ✅      | `ConversationChain` para conversa + pipeline LCEL para saídas estruturadas         |
| Memória gerenciada   | ✅      | `ConversationChain` + `ConversationTokenBufferMemory`, limite de 1200 tokens       |
| Pydantic v2          | ✅      | `schemas.py` — `CorrecaoRedacao` e `RelatorioSessao`                               |
| PydanticOutputParser | ✅      | Integrado às chains de saída estruturada                                           |
| System prompt        | ✅      | `prompts.py` — persona, escopo, regras, segurança e XML tagging                    |
| Domínio documentado  | ✅      | Este README + `prompts.py`                                                         |
| Context engineering  | ✅      | `context_rot.py` — experimento com contexto crescente                              |
| Contagem de tokens   | ✅      | `tiktoken` + tokens reportados pelo Ollama                                         |
| Métricas de contexto | ✅      | qualidade, tokens e latência por janela                                            |
| Meta prompting       | ⏳      | Diferencial opcional não implementado                                              |

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

### Context Rot

```bash
python -m app.context_rot
```

Executa o experimento de contexto crescente e gera:

```text
context_rot_resultados.md
```

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
                         │         Guardrails           │
                         │       guardrails.py          │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
             ┌────────────────────────────────────────────────┐
             │                  Chain 1 — Chat                 │
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
             │              Chain 2 — Estruturada              │
             │                                                │
             │ ChatPromptTemplate                             │
             │          │                                     │
             │          ▼                                     │
             │ ChatOllama                                     │
             │          │                                     │
             │          ▼                                     │
             │ PydanticOutputParser                           │
             │                                                │
             │   ├── CorrecaoRedacao                          │
             │   └── RelatorioSessao                          │
             └────────────────────────────────────────────────┘
```

A Chain 1 é responsável pela conversa do aluno e utiliza memória gerenciada.

A Chain 2 utiliza LCEL para produzir saídas estruturadas e validadas por schemas Pydantic.

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

---

## Estrutura do projeto

```text
app/
├── __init__.py
├── main.py              # Interface Gradio e ponto de entrada
├── chain.py             # Chains LCEL e orquestrador TutorENEM
├── memory_manager.py    # ConversationTokenBufferMemory por sala
├── schemas.py           # Schemas Pydantic v2
├── context_rot.py       # Experimento de contexto crescente
├── prompts.py           # System prompts e configurações das matérias
├── guardrails.py        # Validação e proteção de entrada/saída
└── tokens.py            # Contagem de tokens com tiktoken

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

A escolha segue a recomendação apresentada na Aula 02 para **tutores educacionais e assistentes de estudo**: a memória TokenBuffer mantém uma janela deslizante das mensagens mais recentes, sendo adequada quando as explicações recentes são mais relevantes para a próxima interação.

### Por que TokenBuffer?

Uma sessão de estudo pode possuir muitos turnos. Manter todo o histórico indefinidamente aumenta o contexto enviado ao modelo.

A `ConversationTokenBufferMemory` mantém as mensagens mais recentes até o limite configurado e descarta as mais antigas quando o limite é atingido. Esse comportamento corresponde à janela deslizante apresentada na Aula 02.

O limite escolhido foi de **1200 tokens**, dentro da faixa de **800–1500 tokens** definida no CKP01 e também dentro da faixa recomendada na Aula 02 para esse tipo de aplicação.

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

O projeto possui um experimento específico para analisar o comportamento do modelo quando o contexto cresce.

O experimento mantém:

* o mesmo system prompt;
* a mesma pergunta final;
* o mesmo fato inicial;
* diferentes quantidades de turnos intermediários;
* distratores introduzidos progressivamente.

O fato plantado utilizado no experimento é:

```text
Meu nome é Ana, estou no 3º ano e minha meta em Matemática no ENEM é 780 pontos.
```

A pergunta final verifica se o modelo consegue recuperar corretamente:

```text
Qual é o meu nome e qual é a minha meta de pontos em Matemática?
```

São avaliadas três métricas:

1. Lembrou o nome;
2. Lembrou a meta;
3. Não confundiu o fato com os distratores.

As janelas atualmente testadas são:

```text
0 → 5 → 10 → 15 → 20 turnos de distração
```

Os tokens são medidos com `tiktoken` e também são coletados os tokens reportados pelo Ollama quando disponíveis.

### Resultado do experimento

No experimento atual, **não foi observada degradação de qualidade nas janelas testadas**. Todas as janelas avaliadas apresentaram 100% de qualidade.

Isso é mantido como resultado experimental, sem fabricar uma degradação que não foi observada.

Para reproduzir:

```bash
python -m app.context_rot
```

O resultado é salvo em:

```text
context_rot_resultados.md
```

O experimento também permite observar o crescimento do número de tokens e a latência conforme o contexto aumenta.

---

## Diferencial: métricas de contexto

Além da avaliação de qualidade, o experimento coleta:

* tokens medidos pelo `tiktoken`;
* tokens de entrada reportados pelo Ollama;
* latência da chamada;
* qualidade percentual;
* métricas individuais de recuperação do contexto.

Isso permite comparar não apenas a resposta final, mas também o crescimento do custo de contexto.

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
[ ] Executar experimento de Context Rot
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
