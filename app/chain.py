"""
chain.py — Pipelines do Tutor ENEM (arquitetura de 2 chains da Aula 03).

  Chain 1 — conversa (Aula 02): ConversationChain + ConversationTokenBufferMemory
            (langchain.chains / langchain.memory), literal ao material. A janela
            de memória (≤1200 tokens, memory_manager.LIMITE_TOKENS_MEMORIA) é
            gerenciada pela própria ConversationTokenBufferMemory, contando
            tokens com tiktoken via custom_get_token_ids (tokens.py). Antes
            disso era uma reimplementação manual em LCEL puro, porque
            langchain.chains/langchain.memory não construíam no Python 3.14
            (ambiente atual: Python 3.13.9, import OK — ver memory_manager.py).
  Chain 2 — saída estruturada (Aula 03): prompt | llm_json | PydanticOutputParser
            · correção de redação  → CorrecaoRedacao
            · relatório da sessão  → RelatorioSessao

context_rot.py (Aula 04) monta sua própria chain básica (prompt | llm),
por fora deste módulo: o experimento precisa de janelas de histórico SEM
limite de tokens, de propósito, para demonstrar a degradação — não pode
usar a memória limitada da Chain 1.

A classe TutorENEM junta tudo com os guardrails e é a única coisa que a
interface (main.py) precisa conhecer.
"""
import os

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.guardrails import (
    LIMITE_CARACTERES_REDACAO, verificar_entrada, verificar_saida,
)
from app.memory_manager import LIMITE_TOKENS_MEMORIA, Sala, SalasDeEstudo
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


def _validar_ambiente():
    """Valida as variáveis obrigatórias do ambiente.

    O projeto usa o Ollama local como gateway para os modelos Cloud.
    A autenticação da conta Ollama é feita pelo CLI (`ollama signin`).
    A OLLAMA_API_KEY continua obrigatória no .env conforme o contrato do CKP01.
    """
    if not os.getenv("OLLAMA_API_KEY"):
        raise RuntimeError(
            "OLLAMA_API_KEY não encontrada. "
            "Configure a chave no arquivo .env."
        )


def criar_llm(json_mode=False):
    _validar_ambiente()

    configuracao = {
        "model": MODELO,
        "temperature": 0.2 if json_mode else 0.7,
        "custom_get_token_ids": token_ids,
        "base_url": "http://localhost:11434",
    }

    if json_mode:
        configuracao["format"] = "json"

    return ChatOllama(**configuracao)


# ==============================================================
# Chain 1 — conversa com memória (Aula 02, literal)
# ==============================================================
def criar_chain_conversa(materia: str, llm: ChatOllama) -> ConversationChain:
    """
    ConversationChain + ConversationTokenBufferMemory.
      memory_key="history"   → casa com o MessagesPlaceholder("history") de criar_prompt_chat
      return_messages=True   → o prompt espera uma LISTA de mensagens, não uma string
      max_token_limit=1200   → mesmo limite usado no projeto todo (memory_manager.py)
      input_key/output_key   → nomes que criar_prompt_chat e o guardrail já esperam

    Atenção: o auto-save do ConversationChain (Chain.prep_outputs) grava a
    resposta BRUTA do modelo assim que o predict() termina — ANTES do nosso
    guardrail de saída rodar. TutorENEM.responder() chama
    Sala.registrar_turno() logo em seguida para corrigir a memória sempre
    que o guardrail tiver alterado a resposta (ver memory_manager.py).
    """
    memoria = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=LIMITE_TOKENS_MEMORIA,
        memory_key="history",
        return_messages=True,
    )
    return ConversationChain(
        llm=llm,
        memory=memoria,
        prompt=criar_prompt_chat(materia),
        input_key="input",
        output_key="response",
    )


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
            sala.chain = criar_chain_conversa(materia, self.llm)
        return sala

    def responder(self, sessao: str, materia: str, texto: str) -> str:
        """Fluxo de 1 turno: guardrail de entrada → ConversationChain → guardrail de saída."""
        sala = self._sala(sessao, materia)
        verificacao = verificar_entrada(texto)

        if not verificacao.permitido:
            # Bloqueado: não chama o modelo e NÃO grava na memória
            resposta = verificacao.mensagem
        else:
            # sala.chain.predict() já salva sozinho (verificacao.texto, resposta_bruta)
            # na ConversationTokenBufferMemory (comportamento automático do LangChain).
            bruta = sala.chain.predict(input=verificacao.texto)
            resposta, alterada = verificar_saida(bruta)
            # Se o guardrail alterou a resposta, corrige o que foi salvo automaticamente
            # (ver Sala.registrar_turno) — a memória nunca guarda o que foi barrado/mascarado.
            sala.registrar_turno(verificacao.texto, resposta, alterada)

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