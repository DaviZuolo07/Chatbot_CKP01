"""
context_rot.py — Demonstração de degradação com contexto crescente (Aula 04).

Experimentos:

A — Contexto crescente:
    Mantém o mesmo fato relevante e aumenta a quantidade de distrações.

B1 — Posição do fato:
    Mantém a mesma quantidade de contexto e altera a posição do fato
    relevante entre início, meio e fim.

B2 — Tarefa mais difícil:
    Aumenta o contexto e exige que o modelo integre várias informações
    do perfil do aluno. Existem distratores semanticamente competitivos
    com nomes, metas e dificuldades.

Todos os experimentos usam:

    ChatPromptTemplate | ChatOllama | StrOutputParser

O histórico é injetado diretamente no MessagesPlaceholder, sem memória
limitada, justamente para observar o comportamento quando o contexto cresce.
"""

import argparse
import re
import time
from statistics import mean

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.prompts import criar_prompt_chat
from app.tokens import contar_tokens


MATERIA = "matematica"

# Cada valor abaixo representa uma quantidade de turnos de distração.
# Mais turnos = histórico maior = mais tokens enviados ao modelo.
JANELAS_PADRAO = (0, 5, 10, 15, 20)

ARQUIVO_RESULTADO = "context_rot_resultados.md"


# ============================================================
# EXPERIMENTO A / B1 — FATO ORIGINAL
# ============================================================

FATO_PLANTADO = (
    "Oi! Meu nome é Ana, estou no 3º ano e minha meta em Matemática "
    "no ENEM é 780 pontos."
)

RESPOSTA_FATO = (
    "Prazer, Ana! Meta de 780 pontos anotada. Vamos estudar juntos "
    "para chegar lá. Por onde quer começar?"
)

PERGUNTA_FINAL = (
    "Qual é o meu nome e qual é a minha meta de pontos em Matemática? "
    "Responda em uma frase."
)

NOME_CORRETO = "ana"
META_CORRETA = "780"


# Distratores usados nos experimentos A e B1.
DISTRATORES = [
    (
        "Minha amiga Júlia disse que a meta dela em Matemática é 820 pontos.",
        "820",
    ),
    (
        "Meu primo Pedro quer tirar 650 em Matemática.",
        "650",
    ),
    (
        "A Mariana, da minha sala, está mirando 710 pontos.",
        "710",
    ),
    (
        "Ah, e em Física a minha meta é 700.",
        "700",
    ),
    (
        "O Lucas falou que 760 já está ótimo para ele.",
        "760",
    ),
    (
        "Minha professora disse que a média da escola foi 590.",
        "590",
    ),
]


TEMAS_DISTRACAO = [
    "regra de três composta",
    "juros compostos",
    "função do 2º grau",
    "probabilidade condicional",
    "área de figuras planas",
    "volume de cilindros",
    "média, moda e mediana",
    "progressão aritmética",
    "análise combinatória",
    "escalas em mapas",
    "porcentagem sucessiva",
    "logaritmos",
    "progressão geométrica",
    "trigonometria no triângulo retângulo",
    "função exponencial",
    "desvio padrão",
    "semelhança de triângulos",
    "razão e proporção",
    "gráficos de setores",
    "prismas",
]


def _resposta_longa(tema: str) -> str:
    """Resposta sintética (~250 tokens) para encher o contexto de forma realista."""
    return (
        f"Boa pergunta! Sobre {tema}: o primeiro passo é identificar quais "
        f"grandezas o enunciado fornece e qual ele pede. No ENEM, {tema} "
        "costuma vir contextualizado em situações do cotidiano — contas de "
        "consumo, receitas, mapas, pesquisas de opinião — então leia o "
        "texto-base com calma antes de fazer qualquer conta. Depois, organize "
        "os dados numa pequena tabela, escreva a relação entre eles e só então "
        "calcule. Um erro comum é pular a conversão de unidades ou esquecer "
        "que porcentagens sucessivas se multiplicam em vez de somar. Outro "
        "erro é marcar a primeira alternativa que 'parece' certa: o ENEM "
        "coloca distratores que correspondem exatamente aos erros mais comuns. "
        "Por isso, ao terminar, confira se o resultado faz sentido na situação "
        "(ordem de grandeza, unidade, sinal). Para praticar, resolva três "
        "questões desse assunto cronometrando o tempo, anote onde errou e "
        f"refaça depois de um dia. Quer que eu monte um exercício inédito "
        f"de {tema} no estilo ENEM?"
    )


# ============================================================
# B2 — PERFIL COMPOSTO E DISTRATORES SEMANTICAMENTE COMPETITIVOS
# ============================================================

PERFIL_B2 = [
    (
        "Meu nome é Ana e estou no 3º ano do Ensino Médio.",
        "Entendido, Ana! Vou considerar que você está no 3º ano.",
    ),
    (
        "Minha meta em Matemática no ENEM é alcançar 780 pontos.",
        "Anotado: sua meta em Matemática no ENEM é 780 pontos.",
    ),
    (
        "Eu tenho facilidade em porcentagem, mas tenho dificuldade em probabilidade.",
        "Entendi. Então porcentagem é um ponto de maior domínio e probabilidade "
        "é uma dificuldade atual.",
    ),
    (
        "Quero priorizar o assunto que mais pode ajudar na minha evolução em Matemática.",
        "Perfeito. Vamos considerar suas dificuldades atuais para definir essa prioridade.",
    ),
]


# Cada distrator possui características parecidas com as do perfil da Ana.
# O objetivo é tornar a recuperação uma tarefa de associação, e não apenas
# uma busca literal por um nome ou número.
DISTRATORES_B2 = [
    {
        "nome": "Júlia",
        "meta": "820",
        "facilidade": "probabilidade",
        "dificuldade": "geometria",
    },
    {
        "nome": "Pedro",
        "meta": "650",
        "facilidade": "funções",
        "dificuldade": "probabilidade",
    },
    {
        "nome": "Mariana",
        "meta": "710",
        "facilidade": "geometria",
        "dificuldade": "funções",
    },
    {
        "nome": "Lucas",
        "meta": "760",
        "facilidade": "porcentagem",
        "dificuldade": "estatística",
    },
]


PERGUNTA_FINAL_B2 = (
    "Considerando especificamente o meu perfil, qual assunto de Matemática "
    "eu deveria priorizar para melhorar minha evolução no ENEM e qual é a "
    "minha meta de pontuação? Explique brevemente a relação entre minha "
    "dificuldade e a prioridade escolhida."
)


def construir_historico(
    turnos_distracao: int,
    posicao: str = "inicio",
) -> list:
    """
    Constrói o mesmo volume de contexto, variando apenas a posição
    em que o fato relevante é apresentado.

    posicao:
        - inicio: fato antes das distrações
        - meio: fato aproximadamente no meio
        - fim: fato depois das distrações
    """
    distracoes = []

    for i in range(turnos_distracao):
        tema = TEMAS_DISTRACAO[i % len(TEMAS_DISTRACAO)]
        pergunta = f"Pode me explicar {tema}?"

        if i % 3 == 2:
            pergunta = (
                f"{DISTRATORES[(i // 3) % len(DISTRATORES)][0]} "
                f"Enfim, pode me explicar {tema}?"
            )

        distracoes += [
            HumanMessage(pergunta),
            AIMessage(_resposta_longa(tema)),
        ]

    fato = [
        HumanMessage(FATO_PLANTADO),
        AIMessage(RESPOSTA_FATO),
    ]

    if posicao == "inicio":
        return fato + distracoes

    if posicao == "fim":
        return distracoes + fato

    if posicao == "meio":
        meio = len(distracoes) // 2
        return distracoes[:meio] + fato + distracoes[meio:]

    raise ValueError(
        f"Posição inválida: {posicao}. Use 'inicio', 'meio' ou 'fim'."
    )


def construir_historico_b2(turnos_distracao: int) -> list:
    """
    Constrói o histórico do B2.

    O perfil relevante da Ana é distribuído em quatro momentos diferentes.
    Entre essas informações são inseridos turnos de distração e distratores
    semanticamente semelhantes.

    Quanto maior turnos_distracao, maior o contexto total enviado ao modelo.
    """
    historico = []

    # Distribuímos os turnos de distração entre os quatro blocos
    # que contêm informações relevantes sobre Ana.
    base = turnos_distracao // 4

    distribuicao = [
        base,
        base,
        base,
        turnos_distracao - (3 * base),
    ]

    indice_distrator = 0

    for bloco, quantidade in enumerate(distribuicao):
        # Informação relevante do perfil da Ana.
        pergunta_relevante, resposta_relevante = PERFIL_B2[bloco]

        historico += [
            HumanMessage(pergunta_relevante),
            AIMessage(resposta_relevante),
        ]

        # Distrações após a informação relevante.
        for i in range(quantidade):
            tema = TEMAS_DISTRACAO[
                (i + bloco * 5) % len(TEMAS_DISTRACAO)
            ]

            pergunta = f"Pode me explicar {tema}?"

            # A cada 3 turnos adicionamos um perfil concorrente.
            if i % 3 == 2:
                distrator = DISTRATORES_B2[
                    indice_distrator % len(DISTRATORES_B2)
                ]

                pergunta = (
                    f"Meu nome é {distrator['nome']}, minha meta em Matemática "
                    f"é {distrator['meta']} pontos. Tenho facilidade em "
                    f"{distrator['facilidade']}, mas dificuldade em "
                    f"{distrator['dificuldade']}. "
                    f"Enfim, pode me explicar {tema}?"
                )

                indice_distrator += 1

            historico += [
                HumanMessage(pergunta),
                AIMessage(_resposta_longa(tema)),
            ]

    return historico


# ============================================================
# A / B1 — AVALIAÇÃO ORIGINAL
# ============================================================

def avaliar(resposta: str, turnos: int) -> dict:
    """Critérios objetivos — sem depender de outro LLM como juiz."""
    texto = resposta.lower()

    numeros_distratores = {
        DISTRATORES[(i // 3) % len(DISTRATORES)][1]
        for i in range(turnos)
        if i % 3 == 2
    }

    criterios = {
        # \b = palavra inteira: "semana" e "banana" não contam como "Ana".
        "lembrou_nome": (
            re.search(
                rf"\b{NOME_CORRETO}\b",
                texto,
            )
            is not None
        ),
        "lembrou_meta": META_CORRETA in texto,
        "sem_confusao": not any(
            numero in texto
            for numero in numeros_distratores
        ),
    }

    criterios["qualidade_pct"] = round(
        100 * sum(criterios.values()) / 3
    )

    return criterios


# ============================================================
# B2 — AVALIAÇÃO DA TAREFA COMPOSTA
# ============================================================

def avaliar_b2(resposta: str) -> dict:
    """
    Avaliação objetiva do B2.

    A resposta ideal precisa:
        1. identificar Ana;
        2. recuperar a meta 780;
        3. identificar probabilidade como prioridade;
        4. não atribuir características de outro aluno à Ana.

    Não utiliza outro LLM como juiz.
    """
    texto = resposta.lower()

    criterios = {
        "identificou_ana": (
            re.search(r"\bana\b", texto) is not None
        ),
        "lembrou_meta": (
            re.search(r"\b780\b", texto) is not None
        ),
        "identificou_prioridade": (
            "probabilidade" in texto
        ),
        "sem_confusao": not any(
            re.search(
                rf"\b{re.escape(distrator['nome'].lower())}\b",
                texto,
            )
            for distrator in DISTRATORES_B2
        ),
    }

    criterios["qualidade_pct"] = round(
        100 * sum(criterios.values()) / 4
    )

    return criterios


# ============================================================
# EXPERIMENTO A
# ============================================================

def executar_experimento(
    janelas=JANELAS_PADRAO,
    repeticoes: int = 1,
    llm=None,
) -> list[dict]:
    """Roda o Experimento A em cada janela."""
    from app.chain import criar_llm

    llm = llm or criar_llm()

    prompt = criar_prompt_chat(MATERIA)
    chain_modelo = prompt | llm
    parser = StrOutputParser()

    linhas = []

    for turnos in janelas:
        historico = construir_historico(turnos)

        entrada = {
            "history": historico,
            "input": PERGUNTA_FINAL,
        }

        texto_prompt = "\n".join(
            mensagem.content
            for mensagem in prompt.format_messages(**entrada)
        )

        tokens_tiktoken = contar_tokens(texto_prompt)

        print(
            f"janela: {turnos:>3} turnos de contexto "
            f"(~{tokens_tiktoken} tokens tiktoken no prompt)"
        )

        rodadas = []

        for _ in range(repeticoes):
            inicio = time.perf_counter()

            mensagem = chain_modelo.invoke(entrada)

            latencia = time.perf_counter() - inicio
            resposta = parser.invoke(mensagem)

            uso = getattr(mensagem, "usage_metadata", None) or {}

            rodadas.append(
                {
                    **avaliar(resposta, turnos),
                    "latencia": latencia,
                    "tokens_ollama": uso.get("input_tokens"),
                    "resposta": resposta,
                }
            )

            print(
                f"  {turnos:>3} turnos → "
                f"{tokens_tiktoken} tokens → "
                f"{rodadas[-1]['qualidade_pct']}% qualidade | "
                f"{resposta[:70]!r}"
            )

        tokens_reais = [
            rodada["tokens_ollama"]
            for rodada in rodadas
            if rodada["tokens_ollama"]
        ]

        linhas.append(
            {
                "turnos_distracao": turnos,
                "tokens_tiktoken": tokens_tiktoken,
                "tokens_ollama": (
                    round(mean(tokens_reais))
                    if tokens_reais
                    else None
                ),
                "latencia_s": round(
                    mean(
                        rodada["latencia"]
                        for rodada in rodadas
                    ),
                    2,
                ),
                "lembrou_nome_pct": round(
                    100
                    * mean(
                        rodada["lembrou_nome"]
                        for rodada in rodadas
                    )
                ),
                "lembrou_meta_pct": round(
                    100
                    * mean(
                        rodada["lembrou_meta"]
                        for rodada in rodadas
                    )
                ),
                "sem_confusao_pct": round(
                    100
                    * mean(
                        rodada["sem_confusao"]
                        for rodada in rodadas
                    )
                ),
                "qualidade_pct": round(
                    mean(
                        rodada["qualidade_pct"]
                        for rodada in rodadas
                    )
                ),
                "exemplo_resposta": (
                    rodadas[-1]["resposta"]
                    .replace("\n", " ")
                    [:120]
                ),
            }
        )

    return linhas


# ============================================================
# EXPERIMENTO B1
# ============================================================

def executar_experimento_b1(
    turnos: int = 160,
    posicoes=("inicio", "meio", "fim"),
    repeticoes: int = 3,
    llm=None,
) -> list[dict]:
    """
    Experimento B1:

    Mantém a mesma quantidade de distração e varia somente a posição
    do fato relevante dentro do contexto.
    """
    from app.chain import criar_llm

    llm = llm or criar_llm()

    prompt = criar_prompt_chat(MATERIA)
    chain_modelo = prompt | llm
    parser = StrOutputParser()

    linhas = []

    for posicao in posicoes:
        historico = construir_historico(
            turnos,
            posicao,
        )

        entrada = {
            "history": historico,
            "input": PERGUNTA_FINAL,
        }

        texto_prompt = "\n".join(
            mensagem.content
            for mensagem in prompt.format_messages(**entrada)
        )

        tokens_tiktoken = contar_tokens(texto_prompt)

        print(
            f"\nposição: {posicao:<6} | "
            f"{turnos} turnos | "
            f"~{tokens_tiktoken} tokens tiktoken"
        )

        rodadas = []

        for repeticao in range(repeticoes):
            inicio = time.perf_counter()

            mensagem = chain_modelo.invoke(entrada)

            latencia = time.perf_counter() - inicio
            resposta = parser.invoke(mensagem)

            uso = getattr(mensagem, "usage_metadata", None) or {}

            avaliacao = avaliar(
                resposta,
                turnos,
            )

            rodada = {
                **avaliacao,
                "latencia": latencia,
                "tokens_ollama": uso.get("input_tokens"),
                "resposta": resposta,
            }

            rodadas.append(rodada)

            print(
                f"  repetição {repeticao + 1}/{repeticoes} → "
                f"{avaliacao['qualidade_pct']}% qualidade | "
                f"{resposta[:100]!r}"
            )

        tokens_reais = [
            rodada["tokens_ollama"]
            for rodada in rodadas
            if rodada["tokens_ollama"]
        ]

        linhas.append(
            {
                "posicao": posicao,
                "turnos_distracao": turnos,
                "tokens_tiktoken": tokens_tiktoken,
                "tokens_ollama": (
                    round(mean(tokens_reais))
                    if tokens_reais
                    else None
                ),
                "latencia_s": round(
                    mean(
                        rodada["latencia"]
                        for rodada in rodadas
                    ),
                    2,
                ),
                "lembrou_nome_pct": round(
                    100
                    * mean(
                        rodada["lembrou_nome"]
                        for rodada in rodadas
                    )
                ),
                "lembrou_meta_pct": round(
                    100
                    * mean(
                        rodada["lembrou_meta"]
                        for rodada in rodadas
                    )
                ),
                "sem_confusao_pct": round(
                    100
                    * mean(
                        rodada["sem_confusao"]
                        for rodada in rodadas
                    )
                ),
                "qualidade_pct": round(
                    mean(
                        rodada["qualidade_pct"]
                        for rodada in rodadas
                    )
                ),
                "exemplo_resposta": (
                    rodadas[-1]["resposta"]
                    .replace("\n", " ")
                    [:120]
                ),
            }
        )

    return linhas


# ============================================================
# EXPERIMENTO B2
# ============================================================

def executar_experimento_b2(
    janelas=(0, 20, 40, 80, 160, 320),
    repeticoes: int = 1,
    llm=None,
) -> list[dict]:
    """
    Experimento B2:

    O contexto cresce enquanto a tarefa final exige integração de
    múltiplas informações do perfil da Ana.
    """
    from app.chain import criar_llm

    llm = llm or criar_llm()

    prompt = criar_prompt_chat(MATERIA)
    chain_modelo = prompt | llm
    parser = StrOutputParser()

    linhas = []

    for turnos in janelas:
        historico = construir_historico_b2(
            turnos
        )

        entrada = {
            "history": historico,
            "input": PERGUNTA_FINAL_B2,
        }

        texto_prompt = "\n".join(
            mensagem.content
            for mensagem in prompt.format_messages(**entrada)
        )

        tokens_tiktoken = contar_tokens(
            texto_prompt
        )

        print(
            f"\nB2 | {turnos:>3} turnos de distração "
            f"| ~{tokens_tiktoken} tokens tiktoken"
        )

        rodadas = []

        for repeticao in range(repeticoes):
            inicio = time.perf_counter()

            mensagem = chain_modelo.invoke(
                entrada
            )

            latencia = time.perf_counter() - inicio
            resposta = parser.invoke(
                mensagem
            )

            uso = getattr(
                mensagem,
                "usage_metadata",
                None,
            ) or {}

            avaliacao = avaliar_b2(
                resposta
            )

            rodada = {
                **avaliacao,
                "latencia": latencia,
                "tokens_ollama": uso.get(
                    "input_tokens"
                ),
                "resposta": resposta,
            }

            rodadas.append(rodada)

            print(
                f"  repetição {repeticao + 1}/{repeticoes} → "
                f"{avaliacao['qualidade_pct']}% qualidade | "
                f"{resposta[:120]!r}"
            )

        tokens_reais = [
            rodada["tokens_ollama"]
            for rodada in rodadas
            if rodada["tokens_ollama"]
        ]

        linhas.append(
            {
                "turnos_distracao": turnos,
                "tokens_tiktoken": tokens_tiktoken,
                "tokens_ollama": (
                    round(mean(tokens_reais))
                    if tokens_reais
                    else None
                ),
                "latencia_s": round(
                    mean(
                        rodada["latencia"]
                        for rodada in rodadas
                    ),
                    2,
                ),
                "identificou_ana_pct": round(
                    100
                    * mean(
                        rodada["identificou_ana"]
                        for rodada in rodadas
                    )
                ),
                "lembrou_meta_pct": round(
                    100
                    * mean(
                        rodada["lembrou_meta"]
                        for rodada in rodadas
                    )
                ),
                "identificou_prioridade_pct": round(
                    100
                    * mean(
                        rodada["identificou_prioridade"]
                        for rodada in rodadas
                    )
                ),
                "sem_confusao_pct": round(
                    100
                    * mean(
                        rodada["sem_confusao"]
                        for rodada in rodadas
                    )
                ),
                "qualidade_pct": round(
                    mean(
                        rodada["qualidade_pct"]
                        for rodada in rodadas
                    )
                ),
                "exemplo_resposta": (
                    rodadas[-1]["resposta"]
                    .replace("\n", " ")
                    [:200]
                ),
            }
        )

    return linhas


# ============================================================
# TABELAS
# ============================================================

def tabela_markdown(linhas: list[dict]) -> str:
    colunas = [
        "turnos_distracao",
        "tokens_tiktoken",
        "tokens_ollama",
        "latencia_s",
        "lembrou_nome_pct",
        "lembrou_meta_pct",
        "sem_confusao_pct",
        "qualidade_pct",
    ]

    cabecalho = (
        "| "
        + " | ".join(colunas)
        + " |\n"
        + "|"
        + "---|" * len(colunas)
        + "\n"
    )

    corpo = "".join(
        "| "
        + " | ".join(
            "n/d"
            if linha[coluna] is None
            else str(linha[coluna])
            for coluna in colunas
        )
        + " |\n"
        for linha in linhas
    )

    return cabecalho + corpo


def tabela_markdown_b1(linhas: list[dict]) -> str:
    colunas = [
        "posicao",
        "turnos_distracao",
        "tokens_tiktoken",
        "tokens_ollama",
        "latencia_s",
        "lembrou_nome_pct",
        "lembrou_meta_pct",
        "sem_confusao_pct",
        "qualidade_pct",
    ]

    cabecalho = (
        "| "
        + " | ".join(colunas)
        + " |\n"
        + "|"
        + "---|" * len(colunas)
        + "\n"
    )

    corpo = "".join(
        "| "
        + " | ".join(
            "n/d"
            if linha[coluna] is None
            else str(linha[coluna])
            for coluna in colunas
        )
        + " |\n"
        for linha in linhas
    )

    return cabecalho + corpo


def tabela_markdown_b2(linhas: list[dict]) -> str:
    colunas = [
        "turnos_distracao",
        "tokens_tiktoken",
        "tokens_ollama",
        "latencia_s",
        "identificou_ana_pct",
        "lembrou_meta_pct",
        "identificou_prioridade_pct",
        "sem_confusao_pct",
        "qualidade_pct",
    ]

    cabecalho = (
        "| "
        + " | ".join(colunas)
        + " |\n"
        + "|"
        + "---|" * len(colunas)
        + "\n"
    )

    corpo = "".join(
        "| "
        + " | ".join(
            "n/d"
            if linha[coluna] is None
            else str(linha[coluna])
            for coluna in colunas
        )
        + " |\n"
        for linha in linhas
    )

    return cabecalho + corpo


# ============================================================
# SALVAMENTO DO EXPERIMENTO A
# ============================================================

def salvar_resultados(
    linhas: list[dict],
    repeticoes: int,
) -> str:
    conteudo = (
        "# Context rot — resultados\n\n"
        f"Modelo: gemma4:cloud · matéria: {MATERIA} · "
        f"repetições por janela: {repeticoes}\n\n"
        "Cada linha é a MESMA pergunta final, com o MESMO system prompt, "
        "variando apenas a janela de contexto (turnos_distracao) — e, "
        "portanto, a quantidade de tokens enviada ao modelo "
        "(tokens_tiktoken / tokens_ollama). "
        "qualidade_pct mede o comportamento real do modelo naquela janela.\n\n"
        f"{tabela_markdown(linhas)}\n"
        "## Respostas de exemplo\n\n"
        + "".join(
            f"- **{linha['turnos_distracao']} turnos:** "
            f"{linha['exemplo_resposta']}\n"
            for linha in linhas
        )
    )

    with open(
        ARQUIVO_RESULTADO,
        "w",
        encoding="utf-8",
    ) as arquivo:
        arquivo.write(conteudo)

    return ARQUIVO_RESULTADO


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimento de context rot do Tutor ENEM"
    )

    # Experimento A
    parser.add_argument(
        "--janelas",
        type=int,
        nargs="+",
        default=list(JANELAS_PADRAO),
    )

    parser.add_argument(
        "--repeticoes",
        type=int,
        default=1,
    )

    # Experimento B1
    parser.add_argument(
        "--b1",
        action="store_true",
        help=(
            "Executa o Experimento B1: "
            "posição do fato no contexto."
        ),
    )

    parser.add_argument(
        "--turnos-b1",
        type=int,
        default=160,
    )

    # Experimento B2
    parser.add_argument(
        "--b2",
        action="store_true",
        help=(
            "Executa o Experimento B2: "
            "tarefa composta com contexto crescente."
        ),
    )

    parser.add_argument(
        "--janelas-b2",
        type=int,
        nargs="+",
        default=[0, 20, 40, 80, 160, 320],
    )

    args = parser.parse_args()

    # ========================================================
    # B1
    # ========================================================

    if args.b1:
        print(
            f"Experimento B1 — "
            f"{args.turnos_b1} turnos × "
            f"3 posições × "
            f"{args.repeticoes} repetição(ões)"
        )

        linhas = executar_experimento_b1(
            turnos=args.turnos_b1,
            repeticoes=args.repeticoes,
        )

        print(
            "\n"
            + tabela_markdown_b1(linhas)
        )

        return

    # ========================================================
    # B2
    # ========================================================

    if args.b2:
        print(
            f"Experimento B2 — "
            f"{len(args.janelas_b2)} janelas × "
            f"{args.repeticoes} repetição(ões)"
        )

        linhas = executar_experimento_b2(
            janelas=args.janelas_b2,
            repeticoes=args.repeticoes,
        )

        print(
            "\n"
            + tabela_markdown_b2(linhas)
        )

        print("\nRespostas de exemplo:")

        for linha in linhas:
            print(
                f"- {linha['turnos_distracao']} turnos: "
                f"{linha['exemplo_resposta']}"
            )

        return

    # ========================================================
    # A
    # ========================================================

    print(
        f"Rodando {len(args.janelas)} janelas × "
        f"{args.repeticoes} repetição(ões)..."
    )

    linhas = executar_experimento(
        args.janelas,
        args.repeticoes,
    )

    print(
        "\n"
        + tabela_markdown(linhas)
    )

    print(
        f"Resultados salvos em "
        f"{salvar_resultados(linhas, args.repeticoes)} "
        f"— cole a tabela no README."
    )


if __name__ == "__main__":
    main()