"""
context_rot.py — Demonstração de degradação com contexto crescente (Aula 04).

Experimento: MESMO prompt final, janelas de contexto diferentes.
  1. Turno 0: o aluno planta um fato — nome (Ana) e meta em Matemática (780).
  2. N turnos de "distração": dúvidas de matemática com respostas longas, e
     a cada 3 turnos um DISTRATOR — outro nome e outra nota (ex.: Júlia, 820).
  3. Pergunta final (sempre igual): "Qual é o meu nome e a minha meta?"
  4. Mede-se: tokens da janela (tiktoken e contagem real do Ollama),
     latência e qualidade = lembrou o nome + lembrou a meta + não confundiu
     com os distratores.

A chain é a básica da Aula 01 (prompt | llm | StrOutputParser) com o
histórico injetado direto no MessagesPlaceholder — sem memória limitada,
justamente para mostrar o que acontece quando o contexto só cresce.

Rodar:  python -m app.context_rot                       (0/5/10/15/20 turnos)
        python -m app.context_rot --janelas 0 20 40 80  --repeticoes 3
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
# Cada valor abaixo é uma quantidade de turnos de distração — ou seja, uma
# JANELA DE CONTEXTO diferente. Mais turnos = histórico maior = mais tokens
# enviados ao modelo no mesmo prompt final (ver PERGUNTA_FINAL). É essa
# variação turnos → tokens que o experimento mede contra a qualidade.
JANELAS_PADRAO = (0, 5, 10, 15, 20)
ARQUIVO_RESULTADO = "context_rot_resultados.md"

FATO_PLANTADO = "Oi! Meu nome é Ana, estou no 3º ano e minha meta em Matemática no ENEM é 780 pontos."
RESPOSTA_FATO = "Prazer, Ana! Meta de 780 pontos anotada. Vamos estudar juntos para chegar lá. Por onde quer começar?"
PERGUNTA_FINAL = "Qual é o meu nome e qual é a minha meta de pontos em Matemática? Responda em uma frase."
NOME_CORRETO, META_CORRETA = "ana", "780"

# Distratores: nomes e notas parecidos competindo com o fato plantado
DISTRATORES = [
    ("Minha amiga Júlia disse que a meta dela em Matemática é 820 pontos.", "820"),
    ("Meu primo Pedro quer tirar 650 em Matemática.", "650"),
    ("A Mariana, da minha sala, está mirando 710 pontos.", "710"),
    ("Ah, e em Física a minha meta é 700.", "700"),
    ("O Lucas falou que 760 já está ótimo para ele.", "760"),
    ("Minha professora disse que a média da escola foi 590.", "590"),
]

TEMAS_DISTRACAO = [
    "regra de três composta", "juros compostos", "função do 2º grau", "probabilidade condicional",
    "área de figuras planas", "volume de cilindros", "média, moda e mediana", "progressão aritmética",
    "análise combinatória", "escalas em mapas", "porcentagem sucessiva", "logaritmos",
    "progressão geométrica", "trigonometria no triângulo retângulo", "função exponencial",
    "desvio padrão", "semelhança de triângulos", "razão e proporção", "gráficos de setores", "prismas",
]


def _resposta_longa(tema: str) -> str:
    """Resposta sintética (~250 tokens) para encher o contexto de forma realista."""
    return (
        f"Boa pergunta! Sobre {tema}: o primeiro passo é identificar quais grandezas o enunciado fornece "
        f"e qual ele pede. No ENEM, {tema} costuma vir contextualizado em situações do cotidiano — "
        "contas de consumo, receitas, mapas, pesquisas de opinião — então leia o texto-base com calma antes "
        "de fazer qualquer conta. Depois, organize os dados numa pequena tabela, escreva a relação entre eles "
        "e só então calcule. Um erro comum é pular a conversão de unidades ou esquecer que porcentagens "
        "sucessivas se multiplicam em vez de somar. Outro erro é marcar a primeira alternativa que 'parece' "
        "certa: o ENEM coloca distratores que correspondem exatamente aos erros mais comuns. Por isso, ao "
        "terminar, confira se o resultado faz sentido na situação do problema (ordem de grandeza, unidade, "
        "sinal). Para praticar, resolva três questões desse assunto cronometrando o tempo, anote onde errou e "
        f"refaça depois de um dia. Quer que eu monte um exercício inédito de {tema} no estilo ENEM?"
    )


def construir_historico(turnos_distracao: int) -> list:
    """Fato plantado + N turnos de distração (distrator a cada 3 turnos)."""
    historico = [HumanMessage(FATO_PLANTADO), AIMessage(RESPOSTA_FATO)]
    for i in range(turnos_distracao):
        tema = TEMAS_DISTRACAO[i % len(TEMAS_DISTRACAO)]
        pergunta = f"Pode me explicar {tema}?"
        if i % 3 == 2:
            pergunta = f"{DISTRATORES[(i // 3) % len(DISTRATORES)][0]} Enfim, pode me explicar {tema}?"
        historico += [HumanMessage(pergunta), AIMessage(_resposta_longa(tema))]
    return historico


def avaliar(resposta: str, turnos: int) -> dict:
    """Critérios objetivos (string match) — sem depender de outro LLM como juiz."""
    texto = resposta.lower()
    numeros_distratores = {
        DISTRATORES[(i // 3) % len(DISTRATORES)][1] for i in range(turnos) if i % 3 == 2
    }
    criterios = {
        # \b = palavra inteira: "semana" e "banana" NÃO contam como "Ana"
        "lembrou_nome": re.search(rf"\b{NOME_CORRETO}\b", texto) is not None,
        "lembrou_meta": META_CORRETA in texto,
        "sem_confusao": not any(n in texto for n in numeros_distratores),
    }
    criterios["qualidade_pct"] = round(100 * sum(criterios.values()) / 3)
    return criterios


def executar_experimento(janelas=JANELAS_PADRAO, repeticoes: int = 1, llm=None) -> list[dict]:
    """Roda o mesmo prompt em cada janela e devolve uma linha por janela (médias)."""
    from app.chain import criar_llm  # import local: permite injetar llm fake em testes

    llm = llm or criar_llm()
    prompt = criar_prompt_chat(MATERIA)
    chain_modelo = prompt | llm  # sem parser aqui para ler o uso de tokens do AIMessage
    parser = StrOutputParser()   # prompt | llm | StrOutputParser, em dois passos

    linhas = []
    for turnos in janelas:
        historico = construir_historico(turnos)
        entrada = {"history": historico, "input": PERGUNTA_FINAL}
        texto_prompt = "\n".join(m.content for m in prompt.format_messages(**entrada))
        tokens_tiktoken = contar_tokens(texto_prompt)
        print(f"janela: {turnos:>3} turnos de contexto (~{tokens_tiktoken} tokens tiktoken no prompt)")

        rodadas = []
        for _ in range(repeticoes):
            inicio = time.perf_counter()
            mensagem = chain_modelo.invoke(entrada)
            latencia = time.perf_counter() - inicio
            resposta = parser.invoke(mensagem)
            uso = getattr(mensagem, "usage_metadata", None) or {}
            rodadas.append({**avaliar(resposta, turnos), "latencia": latencia,
                            "tokens_ollama": uso.get("input_tokens"), "resposta": resposta})
            print(f"  {turnos:>3} turnos → {tokens_tiktoken} tokens → "
                  f"{rodadas[-1]['qualidade_pct']}% qualidade | {resposta[:70]!r}")

        tokens_reais = [r["tokens_ollama"] for r in rodadas if r["tokens_ollama"]]
        linhas.append({
            "turnos_distracao": turnos,
            "tokens_tiktoken": tokens_tiktoken,
            "tokens_ollama": round(mean(tokens_reais)) if tokens_reais else None,
            "latencia_s": round(mean(r["latencia"] for r in rodadas), 2),
            "lembrou_nome_pct": round(100 * mean(r["lembrou_nome"] for r in rodadas)),
            "lembrou_meta_pct": round(100 * mean(r["lembrou_meta"] for r in rodadas)),
            "sem_confusao_pct": round(100 * mean(r["sem_confusao"] for r in rodadas)),
            "qualidade_pct": round(mean(r["qualidade_pct"] for r in rodadas)),
            "exemplo_resposta": rodadas[-1]["resposta"].replace("\n", " ")[:120],
        })
    return linhas


def tabela_markdown(linhas: list[dict]) -> str:
    colunas = ["turnos_distracao", "tokens_tiktoken", "tokens_ollama", "latencia_s",
               "lembrou_nome_pct", "lembrou_meta_pct", "sem_confusao_pct", "qualidade_pct"]
    cabecalho = "| " + " | ".join(colunas) + " |\n|" + "---|" * len(colunas) + "\n"
    corpo = "".join(
        "| " + " | ".join("n/d" if l[c] is None else str(l[c]) for c in colunas) + " |\n" for l in linhas
    )
    return cabecalho + corpo


def salvar_resultados(linhas: list[dict], repeticoes: int) -> str:
    conteudo = (
        f"# Context rot — resultados\n\nModelo: gemma4:cloud · matéria: {MATERIA} · "
        f"repetições por janela: {repeticoes}\n\n"
        "Cada linha é a MESMA pergunta final, com o MESMO system prompt, variando apenas "
        "a janela de contexto (turnos_distracao) — e, portanto, a quantidade de tokens "
        "enviada ao modelo (tokens_tiktoken / tokens_ollama). qualidade_pct mede o "
        "comportamento real do modelo naquela janela.\n\n"
        f"{tabela_markdown(linhas)}\n"
        "## Respostas de exemplo\n\n"
        + "".join(f"- **{l['turnos_distracao']} turnos:** {l['exemplo_resposta']}\n" for l in linhas)
    )
    with open(ARQUIVO_RESULTADO, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)
    return ARQUIVO_RESULTADO


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento de context rot do Tutor ENEM")
    parser.add_argument("--janelas", type=int, nargs="+", default=list(JANELAS_PADRAO))
    parser.add_argument("--repeticoes", type=int, default=1)
    args = parser.parse_args()

    print(f"Rodando {len(args.janelas)} janelas × {args.repeticoes} repetição(ões)...")
    linhas = executar_experimento(args.janelas, args.repeticoes)
    print("\n" + tabela_markdown(linhas))
    print(f"Resultados salvos em {salvar_resultados(linhas, args.repeticoes)} — cole a tabela no README.")


if __name__ == "__main__":
    main()