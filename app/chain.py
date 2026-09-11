"""
chain.py — Pipelines LCEL do chatbot.

Etapa 1: criação do ChatOllama (gemma4:cloud via Ollama Cloud).
Etapa 2 (atual): chain básica  prompt | llm | StrOutputParser()  (Aula 01),
usando o system prompt com XML tagging de prompts.py.
Próximas etapas:
  - ConversationChain com memória para o chat            (Aula 02)
  - chain estruturada prompt | llm_json | Pydantic       (Aula 03)
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.prompts import criar_prompt_chat

# Carrega OLLAMA_API_KEY e OLLAMA_HOST do .env para os.environ.
# Precisa acontecer ANTES de instanciar o ChatOllama — senão dá 401 (Aula 01).
load_dotenv()

MODELO = "gemma4:cloud"  # único modelo permitido no CKP01
TEMPERATURA = 0.7        # padrão do curso desde o 1º semestre (Aula 01)


def _validar_ambiente() -> None:
    """Confere se a chave foi configurada e garante o host da Ollama Cloud."""
    chave = os.getenv("OLLAMA_API_KEY", "").strip()
    if not chave or chave == "sua_chave_aqui":
        raise RuntimeError(
            "OLLAMA_API_KEY não configurada. Copie .env.example para .env "
            "e cole a chave gerada em https://ollama.com/settings/keys"
        )
    # Se o .env não trouxer o host, usa o endpoint da Ollama Cloud
    os.environ.setdefault("OLLAMA_HOST", "https://ollama.com")


def criar_llm() -> ChatOllama:
    """
    Instancia o ChatOllama no mesmo padrão das aulas:
    o cliente Ollama lê OLLAMA_HOST e OLLAMA_API_KEY do ambiente
    e envia a chave automaticamente no header de autenticação.
    """
    _validar_ambiente()
    return ChatOllama(model=MODELO, temperature=TEMPERATURA)


def criar_chain_basica(llm: ChatOllama | None = None) -> Runnable:
    """
    Chain stateless da Aula 01:  prompt | llm | StrOutputParser()
      - prompt: ChatPromptTemplate (system com XML + human com {pergunta})
      - llm:    ChatOllama → devolve AIMessage
      - parser: StrOutputParser → devolve str (sem .content)
    Recebe o llm opcionalmente para reaproveitar a mesma instância
    nas próximas chains (memória e Pydantic).
    """
    llm = llm or criar_llm()
    return criar_prompt_chat() | llm | StrOutputParser()
