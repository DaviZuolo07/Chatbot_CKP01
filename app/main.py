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
/* ============================================================
   Skin visual "caderno / editorial" — só CSS, nenhuma lógica.
   Paleta e tipografia espelham o preview_visual_tutor_enem.html.
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --papel:#F6F5F1;
  --papel-linha:#E7E4DB;
  --tinta:#1B2430;
  --tinta-suave:#5B6472;
  --verde-quadro:#1F3D2B;
  --verde-quadro-claro:#2C5641;
  --borda:#DEDBD0;
  --card:#FFFFFF;

  --cor-redacao:#E15A45;
  --cor-matematica:#2F6FED;
  --cor-historia:#C97B2E;
  --cor-biologia:#3F9142;
  --cor-fisica:#7C5CFC;
}

/* Força tema claro sempre — sobrescreve as variáveis internas do Gradio
   (usadas em botões, radio, labels etc.) tanto no modo claro quanto no
   modo escuro do sistema, para nunca cair em "texto branco no fundo branco". */
:root, .dark, .dark * {
  color-scheme: light !important;
  --body-background-fill: #F6F5F1 !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #F6F5F1 !important;
  --block-background-fill: #FFFFFF !important;
  --block-border-color: #DEDBD0 !important;
  --border-color-primary: #DEDBD0 !important;
  --border-color-accent: #DEDBD0 !important;
  --body-text-color: #1B2430 !important;
  --body-text-color-subdued: #5B6472 !important;
  --block-label-text-color: #5B6472 !important;
  --block-title-text-color: #1B2430 !important;
  --input-background-fill: #FBFAF7 !important;
  --checkbox-label-background-fill: #FBFAF7 !important;
  --checkbox-label-background-fill-selected: #FFFFFF !important;
  --checkbox-label-text-color: #1B2430 !important;
  --checkbox-label-text-color-selected: #1B2430 !important;
  --checkbox-background-color: #FFFFFF !important;
  --checkbox-border-color: #DEDBD0 !important;
  --neutral-50:#FFFFFF; --neutral-100:#F6F5F1; --neutral-200:#E7E4DB;
  --neutral-700:#5B6472; --neutral-800:#1B2430; --neutral-900:#1B2430;
}
html, body, gradio-app {
  background: #F6F5F1 !important;
  color: #1B2430 !important;
}
.gradio-container, .dark .gradio-container {
  background: #F6F5F1 !important;
  color: #1B2430 !important;
}
/* qualquer texto solto dentro do app fica preto por padrão... */
.gradio-container, .gradio-container p, .gradio-container span, .gradio-container label,
.gradio-container li, .gradio-container td, .gradio-container th, .gradio-container div {
  color: #1B2430;
}
/* ...exceto onde o fundo é escuro/colorido de propósito (botão primário e
   balão do aluno), que continuam com texto claro para manter contraste. */
#btn-enviar, #btn-enviar *,
.gradio-container button.primary, .gradio-container button.primary *,
#chat .user-row .message, #chat .user-row .message *,
#chat .message.user, #chat .message.user * {
  color: #F6F5F1 !important;
}

/* ---------- aumenta a proporção geral da interface ---------- */
.gradio-container {
  zoom: 1.18;
}

.gradio-container {
  max-width: 1420px !important;
  margin: auto !important;
  background: var(--papel) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--tinta) !important;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
  font-family: 'Fraunces', serif !important;
  letter-spacing: -.01em;
  color: var(--tinta);
}
footer {display: none !important;}

/* ---------- cabeçalho / título ---------- */
#topo-titulo h1 {font-size: 22px !important; font-weight: 600 !important; margin-bottom: 2px !important;}
#topo-titulo p, #topo-titulo * {color: var(--tinta-suave);}

/* ---------- abas superiores (Chat / Redação / Context rot) ---------- */
.tab-nav, div[role="tablist"] {
  border-bottom: 1px solid var(--borda) !important;
  background: transparent !important;
  gap: 4px !important;
}
.tab-nav button, div[role="tablist"] button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  color: var(--tinta-suave) !important;
  border-radius: 10px 10px 0 0 !important;
  border: none !important;
  background: transparent !important;
}
.tab-nav button.selected, div[role="tablist"] button[aria-selected="true"] {
  color: var(--tinta) !important;
  background: var(--card) !important;
  border: 1px solid var(--borda) !important;
  border-bottom: 1px solid var(--card) !important;
  font-weight: 600 !important;
}

/* ---------- sidebar (coluna de matérias) ---------- */
#sidebar {
  background: var(--card);
  border-right: 1px solid var(--borda);
  border-radius: 16px 0 0 16px;
  padding-right: 6px !important;
}

/* Radio "Matéria" viram abas de caderno coloridas, uma por matéria,
   na mesma ordem de app/prompts.py (redação, matemática, história,
   biologia, física) usada em listar_materias(). */
#materia-radio label,
#materia-radio .wrap label {
  position: relative;
  border: 1px solid var(--borda) !important;
  border-right: none !important;
  border-radius: 12px 0 0 12px !important;
  background: #FBFAF7 !important;
  padding: 11px 16px 11px 14px !important;
  margin-bottom: 7px !important;
  box-shadow: none !important;
}
#materia-radio label::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0; width: 4px;
  border-radius: 3px 0 0 3px;
  background: var(--tinta-suave);
}
#materia-radio label:nth-child(1)::before {background: var(--cor-redacao);}
#materia-radio label:nth-child(2)::before {background: var(--cor-matematica);}
#materia-radio label:nth-child(3)::before {background: var(--cor-historia);}
#materia-radio label:nth-child(4)::before {background: var(--cor-biologia);}
#materia-radio label:nth-child(5)::before {background: var(--cor-fisica);}

#materia-radio label.selected,
#materia-radio label[data-selected="true"],
#materia-radio input:checked + span {
  background: var(--card) !important;
}
#materia-radio label:has(input:checked) {
  background: var(--card) !important;
  box-shadow: -1px 2px 10px -4px rgba(27,36,48,.18) !important;
  transform: translateX(6px);
  font-weight: 600;
}
#materia-radio label:nth-child(1):has(input:checked) {border-color: var(--cor-redacao) !important;}
#materia-radio label:nth-child(2):has(input:checked) {border-color: var(--cor-matematica) !important;}
#materia-radio label:nth-child(3):has(input:checked) {border-color: var(--cor-historia) !important;}
#materia-radio label:nth-child(4):has(input:checked) {border-color: var(--cor-biologia) !important;}
#materia-radio label:nth-child(5):has(input:checked) {border-color: var(--cor-fisica) !important;}

#cabecalho-materia {
  padding: 10px 4px 4px 4px;
}
#cabecalho-materia h3 {font-size: 15px !important; margin: 0 0 2px 0 !important;}
#cabecalho-materia small {color: var(--tinta-suave); font-size: 11.5px;}

/* ---------- caixa "memória da sala" ---------- */
#memoria-sala {
  margin-top: 6px;
  padding: 12px 13px;
  background: #FBFAF7;
  border: 1px dashed var(--borda);
  border-radius: 12px;
  font-size: 11.5px;
  color: var(--tinta-suave) !important;
}
#memoria-sala p, #memoria-sala strong {color: var(--tinta-suave) !important; margin: 0;}

/* botões da sidebar */
#btn-limpar, #btn-relatorio {
  font-size: 13px !important;
  font-weight: 500 !important;
  border-radius: 10px !important;
  border: 1px solid var(--borda) !important;
  background: #FBFAF7 !important;
  color: var(--tinta) !important;
  box-shadow: none !important;
}

/* ---------- área de chat ---------- */
#chat-area {background: var(--card); border-radius: 0 16px 16px 0;}
#chat {
  border: 1px solid var(--borda) !important;
  border-radius: 16px !important;
  background-image: repeating-linear-gradient(var(--papel) 0px, var(--papel) 34px, var(--papel-linha) 35px) !important;
  background-size: 100% 35px !important;
  background-color: var(--papel) !important;
}
#chat .message-wrap, #chat .bubble-wrap {background: transparent !important;}

/* balões do tutor (bot) — canto do quadro-verde */
#chat .bot-row .message,
#chat .message.bot,
#chat [data-testid="bot"] .message-content {
  background: #fff !important;
  border: 1px solid var(--borda) !important;
  border-left: 3px solid var(--cor-matematica) !important;
  border-radius: 0 18px 18px 18px !important;
  color: var(--tinta) !important;
}
/* balões do aluno (user) — verde quadro-negro */
#chat .user-row .message,
#chat .message.user,
#chat [data-testid="user"] .message-content {
  background: var(--verde-quadro) !important;
  border-radius: 18px 0 18px 18px !important;
  color: var(--papel) !important;
}
#chat .message.user * {color: var(--papel) !important;}
#chat code {
  background: rgba(47,111,237,.08) !important;
  border-radius: 5px !important;
}

/* ---------- campo de entrada ---------- */
#entrada-wrap {padding-top: 10px;}
#entrada textarea {
  font-size: 15px !important;
  font-family: 'Inter', sans-serif !important;
  border-radius: 14px !important;
  border: 1px solid var(--borda) !important;
  background: #FBFAF7 !important;
}
#btn-enviar {
  background: var(--verde-quadro) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 14px !important;
  font-weight: 600 !important;
}
#btn-enviar:hover {background: var(--verde-quadro-claro) !important;}
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
    with gr.Blocks(
        title=NOME_PRODUTO,
        theme=gr.themes.Soft(primary_hue="indigo", font=[gr.themes.GoogleFont("Inter"), "sans-serif"]),
        css=CSS,
    ) as demo:
        with gr.Row(elem_id="topo-titulo"):
            gr.Markdown(f"# 🎓 {NOME_PRODUTO}\nTutor de ensino médio com foco no ENEM — escolha a matéria e estude.")
        texto_pendente = gr.State("")  # passa a mensagem do passo 1 para o passo 2

        with gr.Tabs():
            # ================= CHAT =================
            with gr.Tab("💬 Chat"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260, elem_id="sidebar"):
                        materia = gr.Radio(listar_materias(), value=MATERIA_INICIAL,
                                           label="Matéria", container=True, elem_id="materia-radio")
                        cabecalho = gr.Markdown(_cabecalho(MATERIA_INICIAL), elem_id="cabecalho-materia")
                        stats = gr.Markdown(elem_id="memoria-sala")
                        btn_limpar = gr.Button("🗑️ Nova conversa nesta matéria", variant="secondary",
                                                elem_id="btn-limpar")
                        btn_relatorio = gr.Button("📋 Gerar relatório da sessão", elem_id="btn-relatorio")
                        with gr.Accordion("Relatório (Pydantic)", open=False) as acc_relatorio:
                            relatorio_md = gr.Markdown()
                            relatorio_json = gr.JSON(label="RelatorioSessao validado")

                    with gr.Column(scale=4, elem_id="chat-area"):
                        chat = gr.Chatbot(
                            elem_id="chat", type="messages", height=580, show_copy_button=True,
                            latex_delimiters=DELIMITADORES_LATEX, show_label=False,
                            placeholder="### Olá! 👋\nMande sua dúvida. Cada matéria tem a sua própria sala e memória.",
                        )
                        with gr.Row(elem_id="entrada-wrap"):
                            entrada = gr.Textbox(elem_id="entrada", placeholder="Digite sua dúvida e aperte Enter…",
                                                 show_label=False, scale=8, autofocus=True, max_lines=6)
                            btn_enviar = gr.Button("Enviar ✏️", variant="primary", scale=1, min_width=90,
                                                    elem_id="btn-enviar")

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