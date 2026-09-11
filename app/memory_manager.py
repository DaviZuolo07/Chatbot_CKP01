"""
memory_manager.py — Memória gerenciada da conversa (Aula 02).

Estratégia escolhida: ConversationTokenBufferMemory com limite de 1200 tokens,
UMA memória por "sala" (sessão do navegador + matéria), tudo em RAM.
Justificativa completa no README (seção "Justificativa da memória").

Por que TokenBuffer e não as outras duas:
  - Buffer: guarda tudo; numa sessão de estudo de 30+ turnos o custo de tokens
    cresce sem limite e o contexto longo degrada a qualidade (ver context_rot.py).
  - Summary: faz uma chamada EXTRA ao LLM por turno e o resumo perde exatamente o
    que o aluno precisa em Matemática/Física — números, contas e fórmulas.
  - TokenBuffer: janela deslizante com teto previsível; mantém literalmente os
    últimos turnos (o exercício em andamento) sem chamada extra.

Demonstração em 6 turnos:  python -m app.memory_manager
"""
import warnings
from dataclasses import dataclass, field
from typing import Any

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models import BaseChatModel

# A Aula 02 usa langchain.memory (0.3.x), que emite aviso de depreciação.
# O aviso é esperado e só polui o terminal — silenciado de propósito.
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
from langchain.memory import ConversationTokenBufferMemory  # noqa: E402

from app.tokens import contar_tokens_mensagens  # noqa: E402

LIMITE_TOKENS_MEMORIA = 1200  # dentro da faixa 800–1500 exigida no CKP01


def criar_memoria(llm: BaseChatModel) -> ConversationTokenBufferMemory:
    """
    Memória de UMA sala. O llm é usado apenas para CONTAR tokens
    (via custom_get_token_ids = tiktoken, configurado em chain.criar_llm).
    return_messages=True → devolve HumanMessage/AIMessage para o
    MessagesPlaceholder("history") do prompt.
    """
    return ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=LIMITE_TOKENS_MEMORIA,
        memory_key="history",
        return_messages=True,
    )


@dataclass
class Sala:
    """Uma sala de estudo = uma matéria dentro de uma sessão do navegador."""
    materia: str
    memoria: ConversationTokenBufferMemory
    chain: Any = None                                   # ConversationChain (criada em chain.py)
    transcricao: list[dict] = field(default_factory=list)  # histórico COMPLETO para a interface

    def estatisticas(self) -> dict:
        """O que a memória está enviando ao modelo agora (≠ transcrição completa)."""
        msgs = self.memoria.chat_memory.messages
        return {
            "mensagens_na_memoria": len(msgs),
            "mensagens_na_conversa": len(self.transcricao),
            "tokens_na_memoria": contar_tokens_mensagens(msgs),
            "limite_tokens": LIMITE_TOKENS_MEMORIA,
        }


class SalasDeEstudo:
    """
    Guarda as salas em RAM: {(id_sessao, materia): Sala}.
    Trocar de matéria não apaga nada — cada matéria tem memória própria,
    e Biologia nunca "vê" o que foi conversado em Física.
    """

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._salas: dict[tuple[str, str], Sala] = {}

    def obter(self, sessao: str, materia: str) -> Sala:
        chave = (sessao, materia)
        if chave not in self._salas:
            self._salas[chave] = Sala(materia=materia, memoria=criar_memoria(self._llm))
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

    print("\n=== load_memory_variables({}) ao final ===")
    for msg in sala.memoria.load_memory_variables({})["history"]:
        print(f"[{msg.type}] {msg.content[:100]}")


if __name__ == "__main__":
    demonstrar_memoria()
