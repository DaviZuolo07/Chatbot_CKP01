"""
main.py — Entry point do Tutor ENEM:  python -m app.main
Sobe a interface Gradio em http://localhost:7860 (enunciado do CKP01).

Abas:
  💬 Chat            — escolhe a matéria na lateral e conversa (chain com memória)
  ✍️ Corrigir redação — chain estruturada → CorrecaoRedacao (Pydantic)
  📉 Context rot      — roda o experimento de degradação e mostra tabela + gráfico

A interface só conhece a classe TutorENEM (chain.py); toda a lógica de
LangChain, memória e guardrails fica fora daqui.
"""
import logging
import sys

import gradio as gr
import pandas as pd

from app.chain import TutorENEM
from app.context_rot import ARQUIVO_RESULTADO, executar_experimento, salvar_resultados
from app.prompts import MATERIAS, NOME_PRODUTO, listar_materias
from app.schemas import CorrecaoRedacao, RelatorioSessao

MATERIA_INICIAL = "matematica"

# Fórmulas: \( \) inline e $$ $$ em bloco. O $ simples fica de fora de
# propósito para "R$ 50,00" não virar fórmula.
DELIMITADORES_LATEX = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#chat {border-radius: 16px;}
#entrada textarea {font-size: 16px;}
footer {display: none !important;}
"""


# --------------------------------------------------------------
# Formatação para a tela
# --------------------------------------------------------------
def _cabecalho(materia: str) -> str:
    cfg = MATERIAS[materia]
    return (f"### {cfg['rotulo']}\n**{cfg['nome_assistente']}** — {cfg['apresentacao']}.\n\n"
            f"<small>{cfg['area_enem']}</small>")


def _stats(tutor: TutorENEM, sessao: str, materia: str) -> str:
    s = tutor.estatisticas(sessao, materia)
    return (f"🧠 **Memória da sala**: {s['tokens_na_memoria']}/{s['limite_tokens']} tokens · "
            f"{s['mensagens_na_memoria']} de {s['mensagens_na_conversa']} mensagens")


def _md_correcao(c: CorrecaoRedacao) -> str:
    linhas = [
        ("C1 · Norma-padrão", c.c1_norma_padrao), ("C2 · Tema e repertório", c.c2_tema_repertorio),
        ("C3 · Argumentação", c.c3_argumentacao), ("C4 · Coesão", c.c4_coesao),
        ("C5 · Proposta de intervenção", c.c5_intervencao),
    ]
    tabela = "| Competência | Nota |\n|---|---|\n" + "".join(f"| {n} | {v} |\n" for n, v in linhas)
    elementos = ", ".join(c.elementos_intervencao) or "nenhum identificado"
    return (
        f"## Nota estimada: {c.nota_total} / 1000\n**Tema:** {c.tema_identificado}\n\n{tabela}\n"
        f"**Elementos da intervenção presentes:** {elementos}\n\n"
        "**✅ Pontos fortes**\n" + "".join(f"- {p}\n" for p in c.pontos_fortes) +
        "\n**🛠️ Pontos a melhorar**\n" + "".join(f"- {p}\n" for p in c.pontos_a_melhorar) +
        (f"\n> {c.comentario_geral}\n" if c.comentario_geral else "") +
        "\n<small>Estimativa gerada por IA para estudo — não é a nota oficial do INEP.</small>"
    )


def _md_relatorio(r: RelatorioSessao) -> str:
    return (
        f"**Nível:** {r.nivel_compreensao} · **Próxima sessão:** {r.tempo_estudo_sugerido_min} min\n\n"
        "**Temas:** " + "; ".join(r.temas_estudados) + "\n\n"
        + ("**Dúvidas:** " + "; ".join(r.duvidas_principais) + "\n\n" if r.duvidas_principais else "")
        + "**Próximos passos:**\n" + "".join(f"1. {p}\n" for p in r.proximos_passos)
        + (f"\n> {r.observacao}" if r.observacao else "")
    )


# --------------------------------------------------------------
# Interface
# --------------------------------------------------------------
def construir_interface(tutor: TutorENEM) -> gr.Blocks:
    with gr.Blocks(title=NOME_PRODUTO, theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
        gr.Markdown(f"# 🎓 {NOME_PRODUTO}\nTutor de ensino médio com foco no ENEM — escolha a matéria e estude.")
        texto_pendente = gr.State("")  # passa a mensagem do passo 1 para o passo 2

        with gr.Tabs():
            # ================= CHAT =================
            with gr.Tab("💬 Chat"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        materia = gr.Radio(listar_materias(), value=MATERIA_INICIAL,
                                           label="Matéria", container=True)
                        cabecalho = gr.Markdown(_cabecalho(MATERIA_INICIAL))
                        stats = gr.Markdown()
                        btn_limpar = gr.Button("🗑️ Nova conversa nesta matéria", variant="secondary")
                        btn_relatorio = gr.Button("📋 Gerar relatório da sessão")
                        with gr.Accordion("Relatório (Pydantic)", open=False) as acc_relatorio:
                            relatorio_md = gr.Markdown()
                            relatorio_json = gr.JSON(label="RelatorioSessao validado")

                    with gr.Column(scale=4):
                        chat = gr.Chatbot(
                            elem_id="chat", type="messages", height=580, show_copy_button=True,
                            latex_delimiters=DELIMITADORES_LATEX, show_label=False,
                            placeholder="### Olá! 👋\nMande sua dúvida. Cada matéria tem a sua própria sala e memória.",
                        )
                        with gr.Row():
                            entrada = gr.Textbox(elem_id="entrada", placeholder="Digite sua dúvida e aperte Enter…",
                                                 show_label=False, scale=8, autofocus=True, max_lines=6)
                            btn_enviar = gr.Button("Enviar", variant="primary", scale=1, min_width=90)

            # ================= REDAÇÃO =================
            with gr.Tab("✍️ Corrigir redação"):
                gr.Markdown("Cole sua redação dissertativa-argumentativa. A correção usa as 5 competências do ENEM "
                            "e é validada pelo schema **CorrecaoRedacao** (Pydantic v2).")
                with gr.Row():
                    with gr.Column(scale=1):
                        tema = gr.Textbox(label="Tema da proposta", placeholder="Ex.: Desafios para a valorização …")
                        redacao = gr.Textbox(label="Redação", lines=18, max_lines=30)
                        btn_corrigir = gr.Button("Corrigir redação", variant="primary")
                    with gr.Column(scale=1):
                        correcao_md = gr.Markdown()
                        with gr.Accordion("JSON validado", open=False):
                            correcao_json = gr.JSON()

            # ================= CONTEXT ROT =================
            with gr.Tab("📉 Context rot"):
                gr.Markdown(
                    "Mesmo prompt final, janelas de contexto crescentes. O aluno diz o nome e a meta no início; "
                    "depois vêm N turnos com **distratores** (outros nomes e notas). Mede-se se o modelo ainda "
                    f"lembra do fato certo. ⚠️ Faz 1 chamada ao modelo por janela e repetição. "
                    f"Resultado salvo em `{ARQUIVO_RESULTADO}`."
                )
                with gr.Row():
                    janelas = gr.Textbox(value="0 5 10 15 20", label="Turnos de distração (separados por espaço)")
                    repeticoes = gr.Slider(1, 3, value=1, step=1, label="Repetições por janela")
                    btn_rot = gr.Button("Rodar experimento", variant="primary")
                tabela_rot = gr.Dataframe(label="Resultados", wrap=True)
                with gr.Row():
                    grafico_qualidade = gr.LinePlot(x="turnos_distracao", y="qualidade_pct",
                                                    title="Qualidade (%) × turnos no contexto", y_lim=[0, 100])
                    grafico_tokens = gr.LinePlot(x="turnos_distracao", y="tokens_tiktoken",
                                                 title="Tokens na janela (tiktoken)")

        # ---------------- Handlers ----------------
        def carregar(m, request: gr.Request):
            sessao = request.session_hash
            return tutor.transcricao(sessao, m), _cabecalho(m), _stats(tutor, sessao, m)

        def passo1_mostrar_pergunta(texto, historico):
            """Mostra a mensagem do aluno na hora (efeito ChatGPT) e limpa a caixa."""
            if not texto.strip():
                return gr.update(), historico, ""
            return "", historico + [{"role": "user", "content": texto}], texto

        def passo2_responder(texto, m, request: gr.Request):
            sessao = request.session_hash
            if not texto.strip():
                return tutor.transcricao(sessao, m), _stats(tutor, sessao, m)
            try:
                tutor.responder(sessao, m, texto)
                historico = tutor.transcricao(sessao, m)
            except Exception as erro:  # rede, chave inválida, limite da API…
                logging.exception("Falha ao responder")
                historico = tutor.transcricao(sessao, m) + [
                    {"role": "user", "content": texto},
                    {"role": "assistant", "content": f"⚠️ Não consegui falar com o modelo agora ({erro}). Tente de novo."},
                ]
            return historico, _stats(tutor, sessao, m)

        def limpar(m, request: gr.Request):
            tutor.limpar(request.session_hash, m)
            return [], _stats(tutor, request.session_hash, m), "", None

        def relatorio(m, request: gr.Request):
            try:
                r = tutor.gerar_relatorio(request.session_hash, m)
                return _md_relatorio(r), r.model_dump(), gr.Accordion(open=True)
            except ValueError as aviso:
                gr.Warning(str(aviso))
            except Exception as erro:
                logging.exception("Falha no relatório")
                gr.Warning(f"O modelo não devolveu um relatório válido: {type(erro).__name__}. Tente de novo.")
            return gr.update(), gr.update(), gr.update()

        def corrigir(t, texto):
            try:
                c = tutor.corrigir_redacao(t, texto)
                return _md_correcao(c), c.model_dump()
            except ValueError as aviso:
                return f"⚠️ {aviso}", None
            except Exception as erro:
                logging.exception("Falha na correção")
                return f"⚠️ O modelo não devolveu uma correção válida ({type(erro).__name__}). Tente de novo.", None

        def rodar_context_rot(txt_janelas, reps):
            try:
                lista = sorted({int(x) for x in txt_janelas.split()})
            except ValueError:
                raise gr.Error("Use apenas números separados por espaço, ex.: 0 5 10 15 20")
            linhas = executar_experimento(lista, int(reps), llm=tutor.llm)
            salvar_resultados(linhas, int(reps))
            df = pd.DataFrame(linhas)
            return df, df, df

        # ---------------- Eventos ----------------
        demo.load(carregar, materia, [chat, cabecalho, stats])
        materia.change(carregar, materia, [chat, cabecalho, stats])

        for gatilho in (entrada.submit, btn_enviar.click):
            gatilho(passo1_mostrar_pergunta, [entrada, chat], [entrada, chat, texto_pendente], queue=False) \
                .then(passo2_responder, [texto_pendente, materia], [chat, stats])

        btn_limpar.click(limpar, materia, [chat, stats, relatorio_md, relatorio_json])
        btn_relatorio.click(relatorio, materia, [relatorio_md, relatorio_json, acc_relatorio])
        btn_corrigir.click(corrigir, [tema, redacao], [correcao_md, correcao_json])
        btn_rot.click(rodar_context_rot, [janelas, repeticoes], [tabela_rot, grafico_qualidade, grafico_tokens])

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # esconde o log de cada requisição
    try:
        tutor = TutorENEM()
    except RuntimeError as erro:  # .env ausente ou chave não configurada
        print(f"❌ {erro}")
        sys.exit(1)

    demo = construir_interface(tutor)
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
