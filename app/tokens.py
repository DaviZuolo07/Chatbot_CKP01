"""
tokens.py — Contagem de tokens com tiktoken (diferencial do CKP01).

Usado em dois lugares:
  - memory_manager.py: a ConversationTokenBufferMemory precisa contar tokens
    para saber quando descartar mensagens antigas;
  - context_rot.py: tamanho de cada janela de contexto no experimento.

Observação: o gemma4 tem tokenizador próprio. O cl100k_base (tiktoken) é uma
APROXIMAÇÃO consistente — serve para comparar janelas entre si. O número real
de tokens do gemma4 é lido do response_metadata do Ollama no context_rot.
"""
from functools import lru_cache

from langchain_core.messages import BaseMessage, get_buffer_string

_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder():
    """Carrega o tokenizador uma única vez. Na 1ª execução o tiktoken baixa
    o arquivo do encoding (precisa de internet); depois fica em cache."""
    try:
        import tiktoken
        return tiktoken.get_encoding(_ENCODING)
    except Exception as erro:  # sem internet / proxy bloqueando
        print(f"⚠️  tiktoken indisponível ({type(erro).__name__}); "
              "usando estimativa de 1 token ≈ 4 caracteres.")
        return None


def token_ids(texto: str) -> list[int]:
    """Lista de IDs de tokens — formato exigido pelo custom_get_token_ids
    do ChatOllama. No modo estimativa, devolve uma lista do tamanho certo."""
    enc = _encoder()
    if enc is not None:
        return enc.encode(texto)
    return list(range(max(1, len(texto) // 4)))


def contar_tokens(texto: str) -> int:
    """Quantidade de tokens de um texto."""
    return len(token_ids(texto))


def contar_tokens_mensagens(mensagens: list[BaseMessage]) -> int:
    """Tokens de uma lista de mensagens (formato 'Human: ... / AI: ...')."""
    return contar_tokens(get_buffer_string(mensagens)) if mensagens else 0
