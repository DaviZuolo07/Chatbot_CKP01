# CKP01 — Chatbot Profissional · [Domínio do grupo]

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**
**Integrantes:** Nome Completo (RM00000) · Nome Completo (RM00000) · Nome Completo (RM00000)
**Peso: 25% · Apresentação: Aula 04 · Entrega: 23:55 do dia da Aula 05 (.zip via Teams — só o líder)**

## Domínio
[Qual domínio, por que foi escolhido, quem são os usuários-alvo]

## Requisitos atendidos
| Requisito | Status | Implementação |
|---|---|---|
| Pipeline LCEL | ⏳ | chain.py |
| ChatOllama | ✅ | gemma4:cloud via Ollama Cloud (.env) |
| Memória gerenciada | ⏳ | memory_manager.py |
| Pydantic v2 (≥4 campos) | ⏳ | schemas.py |
| Context rot | ⏳ | context_rot.py |
| Domínio documentado | ⏳ | Este README + prompts.py |

## Como executar (local — sem Colab)
```bash
cp .env.example .env        # edite com sua OLLAMA_API_KEY — este arquivo NÃO vai no .zip
pip install -r requirements.txt
python -m app.main          # Gradio: http://localhost:7860
```
> Windows: use `copy .env.example .env`. Recomendado usar um ambiente virtual (`python -m venv .venv`).

## Justificativa da memória
[Escolha + por quê + efeito no custo de tokens]
