"""
memory_manager.py — Memória gerenciada da conversa (Aula 02).

Estratégia escolhida: janela deslizante por limite de TOKENS (1200), UMA
memória por "sala" (sessão do navegador + matéria), tudo em RAM. Mesma
estratégia ensinada como ConversationTokenBufferMemory na Aula 02 — a
implementação aqui é manual (lista de BaseMessage podada por tokens.py)
em vez de langchain.memory, porque esse pacote legado (pydantic v1-style)
não constrói mais no Python 3.14. O próprio material da Aula 02 cita
RunnableWithMessageHistory como "a abordagem moderna para memória em LCEL
(0.3+)" — aqui vamos de mais simples ainda: histórico gerenciado à mão e
passado direto pro MessagesPlaceholder("history"), o mesmo padrão que já
usamos em criar_chain_basica / context_rot.py. Justificativa completa da
ESCOLHA de TokenBuffer (vs. Buffer/Summary) está no README.

Por que TokenBuffer e não as outras duas:
  - Buffer: guarda tudo; numa sessão de estudo de 30+ turnos o custo de tokens
    cresce sem limite e o contexto longo degrada a qualidade (ver context_rot.py).
  - Summary: faz uma chamada EXTRA ao LLM por turno e o resumo perde exatamente o
    que o aluno precisa em Matemática/Física — números, contas e fórmulas.
  - TokenBuffer: janela deslizante com teto previsível; mantém literalmente os
    últimos turnos (o exercício em andamento) sem chamada extra.

Demonstração em 6 turnos:  python -m app.memory_manager
"""
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.tokens import contar_tokens_mensagens

LIMITE_TOKENS_MEMORIA = 1200  # dentro da faixa 800–1500 exigida no CKP01


def _podar(mensagens: list[BaseMessage], limite: int) -> list[BaseMessage]:
    """Remove as mensagens MAIS ANTIGAS (início da lista) até a janela
    caber no limite de tokens — o mesmo comportamento do TokenBufferMemory."""
    while len(mensagens) > 1 and contar_tokens_mensagens(mensagens) > limite:
        mensagens.pop(0)
    return mensagens


@dataclass
class Sala:
    """Uma sala de estudo = uma matéria dentro de uma sessão do navegador."""
    materia: str
    chain: Any = None                                       # Runnable (criada em chain.py)
    historico: list[BaseMessage] = field(default_factory=list)  # janela ≤1200 tokens, vai pro modelo
    transcricao: list[dict] = field(default_factory=list)       # histórico COMPLETO para a interface

    def registrar_turno(self, pergunta_sanitizada: str, resposta: str) -> None:
        """Grava o turno na janela de memória e poda pelo limite de tokens."""
        self.historico += [HumanMessage(pergunta_sanitizada), AIMessage(resposta)]
        self.historico = _podar(self.historico, LIMITE_TOKENS_MEMORIA)

    def estatisticas(self) -> dict:
        """O que a memória está enviando ao modelo agora (≠ transcrição completa)."""
        return {
            "mensagens_na_memoria": len(self.historico),
            "mensagens_na_conversa": len(self.transcricao),
            "tokens_na_memoria": contar_tokens_mensagens(self.historico),
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