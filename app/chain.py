"""
chain.py — Pipelines LCEL do Tutor ENEM (arquitetura de 2 chains da Aula 03).

  Chain 1 — conversa (Aula 02): prompt | llm | StrOutputParser, com o
            histórico da sala (janela ≤1200 tokens, memory_manager.py)
            passado manualmente no MessagesPlaceholder("history") a cada
            invoke — mesmo conceito de TokenBufferMemory ensinado na Aula
            02, sem depender de langchain.chains/langchain.memory (esse
            pacote legado não constrói no Python 3.14; ver memory_manager.py).
  Chain 2 — saída estruturada (Aula 03):  prompt | llm_json | PydanticOutputParser
            · correção de redação  → CorrecaoRedacao
            · relatório da sessão  → RelatorioSessao
  Chain básica (Aula 01): prompt | llm | StrOutputParser — usada no context_rot
            e agora também na Chain 1 (mesmo formato, histórico manual).

A classe TutorENEM junta tudo com os guardrails e é a única coisa que a
interface (main.py) precisa conhecer.
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.guardrails import (
    LIMITE_CARACTERES_REDACAO, verificar_entrada, verificar_saida,
)
from app.memory_manager import Sala, SalasDeEstudo
from app.prompts import (
    CORRECAO_HUMAN, CORRECAO_SYSTEM, MATERIAS, RELATORIO_HUMAN, RELATORIO_SYSTEM,
    criar_prompt_chat,
)
from app.schemas import CorrecaoRedacao, RelatorioSessao
from app.tokens import token_ids

# Carrega OLLAMA_API_KEY e OLLAMA_HOST do .env para os.environ.
# Precisa acontecer ANTES de instanciar o ChatOllama — senão dá 401 (Aula 01).
load_dotenv()

MODELO = "gemma4:cloud"      # único modelo permitido no CKP01
TEMPERATURA_CHAT = 0.7       # padrão do curso (Aula 01): respostas didáticas e naturais
TEMPERATURA_JSON = 0.2       # extração/correção: queremos consistência, não criatividade
MIN_PALAVRAS_REDACAO = 60    # abaixo disso não é redação — nem chama o modelo
MAX_CARACTERES_HISTORICO = 8000  # recorte da conversa enviado ao relatório


def _validar_ambiente() -> None:
    """Confere se a chave foi configurada e garante o host da Ollama Cloud."""
    chave = os.getenv("OLLAMA_API_KEY", "").strip()
    if not chave or chave == "sua_chave_aqui":
        raise RuntimeError(
            "OLLAMA_API_KEY não configurada. Copie .env.example para .env "
            "e cole a chave gerada em https://ollama.com/settings/keys"
        )
    os.environ.setdefault("OLLAMA_HOST", "https://ollama.com")


def criar_llm(json_mode: bool = False) -> ChatOllama:
    """
    ChatOllama no padrão das aulas (host e chave lidos do ambiente).
    - custom_get_token_ids: a TokenBufferMemory conta tokens com tiktoken
      (sem isso o LangChain tentaria baixar o tokenizador GPT-2 via transformers).
    - json_mode: format="json" do Ollama + PydanticOutputParser (Aula 03).
    """
    _validar_ambiente()
    if json_mode:
        return ChatOllama(model=MODELO, temperature=TEMPERATURA_JSON, format="json",
                          custom_get_token_ids=token_ids)
    return ChatOllama(model=MODELO, temperature=TEMPERATURA_CHAT,
                      custom_get_token_ids=token_ids)


# ==============================================================
# Chain básica (Aula 01) — stateless, histórico passado manualmente
# ==============================================================
def criar_chain_basica(materia: str, llm: ChatOllama) -> Runnable:
    """prompt | llm | StrOutputParser()  →  invoke({"history": [...], "input": "..."})"""
    return criar_prompt_chat(materia) | llm | StrOutputParser()


# ==============================================================
# Chain 1 — conversa com memória (Aula 02)
# ==============================================================
# criar_chain_conversa foi removida: a Chain 1 agora É a chain básica
# (criar_chain_basica), invocada com o histórico da Sala passado à mão
# em {"history": sala.historico, "input": texto} — ver TutorENEM.responder.
# Isso substitui a antiga ConversationChain(llm, memory, prompt), que
# dependia de langchain.chains (quebra no Python 3.14).


# ==============================================================
# Chain 2 — saída estruturada (Aula 03)
# ==============================================================
def criar_chain_correcao(llm_json: ChatOllama) -> Runnable:
    """prompt | llm_json | PydanticOutputParser(CorrecaoRedacao)"""
    parser = PydanticOutputParser(pydantic_object=CorrecaoRedacao)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CORRECAO_SYSTEM),
        ("human", CORRECAO_HUMAN),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm_json | parser


def criar_chain_relatorio(llm_json: ChatOllama) -> Runnable:
    """prompt | llm_json | PydanticOutputParser(RelatorioSessao)"""
    parser = PydanticOutputParser(pydantic_object=RelatorioSessao)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RELATORIO_SYSTEM),
        ("human", RELATORIO_HUMAN),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm_json | parser


# ==============================================================
# Orquestrador — o que a interface usa
# ==============================================================
class TutorENEM:
    def __init__(self, llm: ChatOllama | None = None, llm_json: ChatOllama | None = None):
        # llm e llm_json podem ser injetados (ex.: modelo fake em testes)
        self.llm = llm or criar_llm()
        self.llm_json = llm_json or criar_llm(json_mode=True)
        self.salas = SalasDeEstudo()
        self.chain_correcao = criar_chain_correcao(self.llm_json)
        self.chain_relatorio = criar_chain_relatorio(self.llm_json)

    def _sala(self, sessao: str, materia: str) -> Sala:
        """Obtém a sala e cria a chain dela na primeira vez."""
        if materia not in MATERIAS:
            raise ValueError(f"Matéria desconhecida: {materia}")
        sala = self.salas.obter(sessao, materia)
        if sala.chain is None:
            sala.chain = criar_chain_basica(materia, self.llm)
        return sala

    def responder(self, sessao: str, materia: str, texto: str) -> str:
        """Fluxo de 1 turno: guardrail de entrada → chain com memória → guardrail de saída."""
        sala = self._sala(sessao, materia)
        verificacao = verificar_entrada(texto)

        if not verificacao.permitido:
            # Bloqueado: não chama o modelo e NÃO grava na memória
            resposta = verificacao.mensagem
        else:
            bruta = sala.chain.invoke({"history": sala.historico, "input": verificacao.texto})
            resposta, _ = verificar_saida(bruta)
            # Grava no histórico a pergunta SANITIZADA (sem dado pessoal/injeção)
            # e a resposta já validada pelo guardrail de saída.
            sala.registrar_turno(verificacao.texto, resposta)

        sala.transcricao += [
            {"role": "user", "content": texto},
            {"role": "assistant", "content": resposta},
        ]
        return resposta

    def transcricao(self, sessao: str, materia: str) -> list[dict]:
        return list(self._sala(sessao, materia).transcricao)

    def estatisticas(self, sessao: str, materia: str) -> dict:
        return self._sala(sessao, materia).estatisticas()

    def limpar(self, sessao: str, materia: str) -> None:
        self.salas.limpar(sessao, materia)

    def corrigir_redacao(self, tema: str, redacao: str) -> CorrecaoRedacao:
        """Chain estruturada 1. Lança ValueError com mensagem amigável se o texto não serve."""
        verificacao = verificar_entrada(redacao, limite=LIMITE_CARACTERES_REDACAO)
        if not verificacao.permitido:
            raise ValueError(verificacao.mensagem)
        if len(verificacao.texto.split()) < MIN_PALAVRAS_REDACAO:
            raise ValueError(f"O texto tem menos de {MIN_PALAVRAS_REDACAO} palavras — escreva a redação completa.")
        tema_ok = verificar_entrada(tema or "Tema não informado")
        return self.chain_correcao.invoke({
            "tema": tema_ok.texto if tema_ok.permitido else "Tema não informado",
            "redacao": verificacao.texto,
        })

    def gerar_relatorio(self, sessao: str, materia: str) -> RelatorioSessao:
        """Chain estruturada 2 — usa a transcrição COMPLETA da sala."""
        sala = self._sala(sessao, materia)
        if not sala.transcricao:
            raise ValueError("Converse um pouco com o tutor antes de gerar o relatório.")
        historico = "\n".join(
            f"{'Aluno' if m['role'] == 'user' else 'Tutor'}: {m['content']}" for m in sala.transcricao
        )[-MAX_CARACTERES_HISTORICO:]
        return self.chain_relatorio.invoke({
            "materia": materia,
            "nome_materia": MATERIAS[materia]["nome_materia"],
            "historico": historico,
        })