# CKP01 — Chatbot Profissional · Tutor ENEM (Ensino Médio)

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**
**Integrantes:** Nome Completo (RM00000) · Nome Completo (RM00000) · Nome Completo (RM00000)
**Peso: 25% · Apresentação: Aula 04 · Entrega: 23:55 do dia da Aula 05 (.zip via Teams — só o líder)**

## Domínio

**Tutor ENEM** — assistente de estudos para o ensino médio com foco no ENEM, organizado em
**5 salas de matéria (lista fechada)**. O aluno escolhe a matéria e conversa com um tutor
especializado nela; cada sala tem persona, escopo, regras e memória próprios.

| Matéria | Tutor(a) | Área do ENEM |
|---|---|---|
| ✍️ Redação | Clara | Redação (0–1000, 5 competências) |
| 📐 Matemática | Teo | Matemática e suas Tecnologias |
| 🌎 História e Geografia | Helena | Ciências Humanas |
| 🧬 Biologia | Bia | Ciências da Natureza |
| ⚡ Física | Max | Ciências da Natureza |

**Por que este domínio:** o ENEM tem matriz de referência pública e estável, e há muitos
documentos oficiais (provas e gabaritos do INEP, Cartilha do Participante da redação, BNCC).
Esses documentos serão a base do RAG no CKP02, com uma coleção por matéria, e o tutor vira
tool do agente no CKP03.

**Usuários-alvo:** estudantes do ensino médio (em geral de 14 a 18 anos) e pessoas que vão
prestar o ENEM. Como muitos são menores de idade, o system prompt exige linguagem adequada à
idade e proíbe pedir dados pessoais.

## Requisitos atendidos
| Requisito | Status | Implementação |
|---|---|---|
| Pipeline LCEL | ✅ | chain.py — `prompt \| llm_json \| PydanticOutputParser()` (correção e relatório) · `prompt \| llm \| StrOutputParser()` (context rot) |
| ChatOllama | ✅ | gemma4:cloud via Ollama Cloud (.env) |
| ChatPromptTemplate | ✅ | prompts.py — system e human separados, variáveis via `.partial()`, sem f-string |
| Memória gerenciada | ✅ | ConversationChain + ConversationTokenBufferMemory (1200 tokens), uma por sala — memory_manager.py |
| Pydantic v2 (≥4 campos) | ✅ | schemas.py — `CorrecaoRedacao` (11 campos) e `RelatorioSessao` (7 campos), com `@field_validator` e `@model_validator` |
| Context rot | ✅ | context_rot.py — janelas de 0/5/10/15/20 turnos, tokens com tiktoken, tabela + gráfico |
| System prompt com persona | ✅ | prompts.py — XML tagging, persona por matéria, sandwich defense |
| Domínio documentado | ✅ | Este README + prompts.py |
| Diferencial: métricas de contexto | ✅ | tiktoken + tokens reais do Ollama + qualidade (%) por janela |
| Diferencial: meta prompting | ⏳ | — |

## Como executar (local — sem Colab)
```bash
cp .env.example .env        # edite com sua OLLAMA_API_KEY — este arquivo NÃO vai no .zip
pip install -r requirements.txt
python -m app.main          # Gradio: http://localhost:7860
```
> Windows: use `copy .env.example .env`. Recomendado usar um ambiente virtual (`python -m venv .venv`).
> Na primeira execução o tiktoken baixa o tokenizador (precisa de internet).

Comandos extras (evidências para a avaliação):
```bash
python -m app.memory_manager   # memória funcionando em 6 turnos (load_memory_variables)
python -m app.context_rot      # experimento de context rot → context_rot_resultados.md
python -m app.guardrails       # casos de teste dos guardrails (ataques e falsos positivos)
```

## Arquitetura (2 chains — Aula 03)
```
                    ┌──────────── guardrails.py (igual para todas as matérias) ────────────┐
aluno ─► Gradio ─►  │ entrada: tamanho · regex anti-injection · tags falsas · mascara CPF/e-mail │
 (main.py)          └────────────────────────────────┬──────────────────────────────────────┘
                                                     ▼
            Chain 1 (chat) — uma por sala (sessão + matéria)
            ConversationChain( ChatPromptTemplate[system da matéria + {history} + {input}],
                               ChatOllama gemma4:cloud, TokenBufferMemory 1200 )
                                                     ▼
                              guardrail de saída: bloqueia vazamento do system prompt

            Chain 2 (estruturada):  prompt | ChatOllama(format="json") | PydanticOutputParser
              · aba "Corrigir redação"  → CorrecaoRedacao
              · botão "Relatório"       → RelatorioSessao
```
O system prompt tem uma **parte fixa** (identidade, público, regras gerais, segurança, formato) e
uma **parte por matéria** (persona, escopo, regras e exemplos few-shot). Cada matéria gera o seu
próprio `ChatPromptTemplate` via `.partial()`. Assim, Biologia só responde Biologia e manda o aluno
trocar de sala se a pergunta for de outra matéria.

## Estrutura
```
app/
├── main.py            # Interface Gradio + entry point
├── chain.py           # Pipelines LCEL + orquestrador TutorENEM
├── memory_manager.py  # TokenBufferMemory por sala + demonstração 6 turnos
├── schemas.py         # Pydantic v2 (CorrecaoRedacao, RelatorioSessao)
├── context_rot.py     # Demonstração de degradação
├── prompts.py         # System prompt XML + 5 matérias + templates estruturados
├── guardrails.py      # Regex anti-injection, dados pessoais, validação de saída
└── tokens.py          # Contagem de tokens (tiktoken)
```

## Justificativa da memória

**Escolha: `ConversationTokenBufferMemory` com `max_token_limit=1200`, uma memória por sala.**

- **Por quê:** uma sessão de estudo é longa (30+ turnos), mas o que importa para a próxima resposta
  é o exercício em andamento, ou seja, os últimos turnos, *literalmente*. A TokenBuffer mantém
  exatamente isso, numa janela deslizante.
- **Por que não Buffer:** o custo cresce sem limite. Com cerca de 300 tokens por turno (pergunta +
  resposta didática), no turno 30 o histórico passaria de ~9.000 tokens por chamada. O experimento
  de context rot mostra que contexto longo também custa qualidade.
- **Por que não Summary:** faz uma chamada extra ao LLM a cada turno e o resumo perde números,
  contas e fórmulas, que são justamente o que importa em Matemática e Física.
- **Efeito no custo de tokens:** cada chamada fica limitada a ≈ system prompt (~1.200 tokens) +
  memória (≤ 1.200) + pergunta. O teto é previsível, independente do tamanho da sessão.
- **Trade-off assumido:** fatos ditos no início da sessão saem da janela depois de alguns turnos.
  Para compensar, a interface guarda a transcrição completa, e o relatório da sessão usa essa
  transcrição completa, não a memória.
- **Uma memória por matéria:** trocar de sala não mistura contexto. O que foi dito em Física
  nunca entra no prompt de Biologia.

## Guardrails (defesa em camadas)
1. **Validação de entrada** (`guardrails.py`): limite de tamanho, regex de prompt injection (ignorar
   instruções, revelar system prompt, troca de identidade, modo irrestrito, tags falsas, blocos base64)
   e mascaramento de CPF, e-mail e telefone (LGPD). Mensagem bloqueada não chama o modelo nem entra na memória.
2. **Separação dados ↔ instruções:** a pergunta vai dentro de `<pergunta_usuario>`; a redação, dentro de `<redacao>`.
3. **System prompt robusto:** seção `<seguranca>` + `<lembrete_final>` (sandwich defense).
4. **Validação de saída:** se a resposta contém tags internas do prompt, é substituída e corrigida na memória.
5. **Logging** de todo bloqueio no terminal.

Os padrões foram testados contra falsos positivos típicos do domínio, como "o governo ignorou as
regras da Constituição" (História) e "aja como examinador" (pedido legítimo).

## Context rot
Mesmo prompt final ("qual é o meu nome e a minha meta?") com 0, 5, 10, 15 e 20 turnos de
distração entre o fato plantado e a pergunta. A cada 3 turnos entra um **distrator** (outro nome e
outra nota). A qualidade é medida por critérios objetivos: lembrou o nome, lembrou a meta e não
confundiu com os distratores.

**Resultados:** rode `python -m app.context_rot` (ou use a aba "📉 Context rot") e cole aqui a
tabela gerada em `context_rot_resultados.md`.

| turnos_distracao | tokens_tiktoken | tokens_ollama | latencia_s | qualidade_pct |
|---|---|---|---|---|
| — | — | — | — | — |
