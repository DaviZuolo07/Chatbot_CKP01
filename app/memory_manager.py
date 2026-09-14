"""
memory_manager.py — Memória gerenciada da conversa (Aula 02).

Estratégia: ConversationTokenBufferMemory (langchain.memory) — janela
deslizante por limite de TOKENS (1200), UMA memória por "sala" (sessão do
navegador + matéria). Literal ao que a Aula 02 pede. Antes disso era uma
reimplementação manual (lista de BaseMessage podada à mão), porque
langchain.chains/langchain.memory não construíam no Python 3.14. No
ambiente atual (Python 3.13.9) o import funciona — ver chain.py, função
criar_chain_conversa.

Por que TokenBuffer e não as outras duas:
  - Buffer: guarda tudo; numa sessão de estudo de 30+ turnos o custo de tokens
    cresce sem limite e o contexto longo degrada a qualidade (ver context_rot.py).
  - Summary: faz uma chamada EXTRA ao LLM por turno e o resumo perde exatamente o
    que o aluno precisa em Matemática/Física — números, contas e fórmulas.
  - TokenBuffer: janela deslizante com teto previsível; mantém literalmente os
    últimos turnos (o exercício em andamento) sem chamada extra.

Detalhe importante (defesa em camadas, ver guardrails.py): o
ConversationChain salva sozinho, automaticamente, a resposta BRUTA do
modelo (comportamento do LangChain — Chain.prep_outputs chama
memory.save_context logo após a chamada ao LLM, ANTES do nosso guardrail
de saída rodar). Sala.registrar_turno() corrige isso: se o guardrail de
saída alterou a resposta (ex.: mascarou um vazamento do system prompt), o
par bruto salvo automaticamente é substituído pelo par final — a memória
NUNCA guarda algo que o guardrail decidiu barrar. Ver TutorENEM.responder
em chain.py.

Demonstração em 6 turnos:  python -m app.memory_manager
"""
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

LIMITE_TOKENS_MEMORIA = 1200  # dentro da faixa 800–1500 exigida no CKP01


@dataclass
class Sala:
    """Uma sala de estudo = uma matéria dentro de uma sessão do navegador."""
    materia: str
    chain: Any = None                                     # ConversationChain (criada em chain.py)
    transcricao: list[dict] = field(default_factory=list)  # histórico COMPLETO para a interface

    @property
    def historico(self) -> list[BaseMessage]:
        """Janela de memória atual (≤1200 tokens) — o que vai pro modelo agora.
        Lida direto da ConversationTokenBufferMemory da chain desta sala."""
        return self.chain.memory.buffer if self.chain is not None else []

    def registrar_turno(
        self, pergunta_sanitizada: str, resposta_final: str, guardrail_alterou: bool,
    ) -> None:
        """
        Chamar DEPOIS de sala.chain.predict(...) e do guardrail de saída.

        O ConversationChain já salvou sozinho (pergunta_sanitizada, resposta_BRUTA)
        durante o predict(). Se o guardrail de saída não alterou nada, não há o
        que corrigir. Se alterou, removemos o par bruto e regravamos com a
        resposta final — save_context já poda pelo limite de tokens de novo.
        """
        if not guardrail_alterou:
            return
        memoria = self.chain.memory
        memoria.chat_memory.messages = memoria.chat_memory.messages[:-2]
        memoria.save_context({"input": pergunta_sanitizada}, {"response": resposta_final})

    def estatisticas(self) -> dict:
        """
        O que a memória está enviando ao modelo agora (≠ transcrição completa).

        tokens_na_memoria usa o MESMO método de contagem que a
        ConversationTokenBufferMemory usa para decidir a poda
        (llm.get_num_tokens_from_messages — soma por mensagem), e não
        tokens.contar_tokens_mensagens (que soma o buffer inteiro como uma
        string só). Os dois métodos podem divergir por 1-2 tokens; usar a
        mesma fonte de verdade da poda evita o painel mostrar um número
        acima do limite quando a memória já está correta.
        """
        buffer = self.historico
        tokens = self.chain.memory.llm.get_num_tokens_from_messages(buffer) if self.chain else 0
        return {
            "mensagens_na_memoria": len(buffer),
            "mensagens_na_conversa": len(self.transcricao),
            "tokens_na_memoria": tokens,
            "limite_tokens": LIMITE_TOKENS_MEMORIA,
        }


class SalasDeEstudo:
    """
    Guarda as salas em RAM: {(id_sessao, materia): Sala}.
    Trocar de matéria não apaga nada — cada matéria tem memória própria,
    e Biologia nunca "vê" o que foi conversado em Física.
    """

    def __init__(self) -> None:
        self._salas: dict[tuple[str, str], Sala] = {}

    def obter(self, sessao: str, materia: str) -> Sala:
        chave = (sessao, materia)
        if chave not in self._salas:
            self._salas[chave] = Sala(materia=materia)
        return self._salas[chave]

    def limpar(self, sessao: str, materia: str) -> None:
        self._salas.pop((sessao, materia), None)


# ==============================================================
# Demonstração exigida no CKP01: memória funcionando em ≥5 turnos.
# ==============================================================
TURNOS_DEMO = [
    "Oi! Meu nome é Ana e estou no 3º ano do ensino médio.",
    "Minha maior dificuldade em Matemática é porcentagem.",
    "Quanto é 15% de 80?",
    "E se eu aumentar esse resultado em 50%?",
    "Qual eu disse que era a minha maior dificuldade?",
    "Você lembra meu nome e em que ano eu estou?",
]


def demonstrar_memoria() -> None:
    from app.chain import TutorENEM  # import local evita import circular

    tutor = TutorENEM()
    sessao, materia = "demo", "matematica"
    for i, pergunta in enumerate(TURNOS_DEMO, start=1):
        resposta = tutor.responder(sessao, materia, pergunta)
        sala = tutor.salas.obter(sessao, materia)
        print(f"\n=== Turno {i} ===\n👤 {pergunta}\n🤖 {resposta}")
        print(f"📊 {sala.estatisticas()}")

    print("\n=== Janela de memória (historico) ao final ===")
    for msg in sala.historico:
        print(f"[{msg.type}] {msg.content[:100]}")


if __name__ == "__main__":
    demonstrar_memoria()