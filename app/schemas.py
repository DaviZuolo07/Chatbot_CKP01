"""
schemas.py — Schemas Pydantic v2 da saída estruturada (Aula 03).

Dois schemas, cada um validado por um PydanticOutputParser no fim de uma
chain LCEL  prompt | llm_json | parser  (ver chain.py):

  - CorrecaoRedacao: correção da redação pelas 5 competências do ENEM.
  - RelatorioSessao: relatório da sessão de estudo de qualquer matéria.

Os @field_validator corrigem desvios comuns do LLM (acentos, maiúsculas,
nota fora da escala) e o @model_validator garante consistência entre
campos (nota total = soma das competências).
"""
import unicodedata
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

MateriaENEM = Literal["redacao", "matematica", "historia_geografia", "biologia", "fisica"]
ElementoIntervencao = Literal["agente", "acao", "modo_meio", "finalidade", "detalhamento"]

NOTAS_VALIDAS = (0, 40, 80, 120, 160, 200)


def _slug(texto: str) -> str:
    """'História e Geografia' → 'historia_e_geografia' (sem acento, minúsculo)."""
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return "_".join(sem_acento.lower().replace("/", " ").split())


def _limpar_lista(itens: list) -> list[str]:
    """Remove itens vazios e duplicados, preservando a ordem."""
    vistos, saida = set(), []
    for item in itens or []:
        texto = str(item).strip()
        if texto and texto.lower() not in vistos:
            vistos.add(texto.lower())
            saida.append(texto)
    return saida


# ==============================================================
# 1) Correção de redação — exclusivo da sala de Redação
# ==============================================================
class CorrecaoRedacao(BaseModel):
    tema_identificado: str = Field(min_length=3, description="Tema da redação, em 1 frase")
    c1_norma_padrao: int = Field(description="Competência 1 (norma-padrão): 0, 40, 80, 120, 160 ou 200")
    c2_tema_repertorio: int = Field(description="Competência 2 (tema, tipo textual e repertório): 0 a 200, múltiplo de 40")
    c3_argumentacao: int = Field(description="Competência 3 (seleção e organização dos argumentos): 0 a 200, múltiplo de 40")
    c4_coesao: int = Field(description="Competência 4 (coesão textual): 0 a 200, múltiplo de 40")
    c5_intervencao: int = Field(description="Competência 5 (proposta de intervenção): 0 a 200, múltiplo de 40")
    nota_total: int = Field(0, description="Soma das 5 competências (0 a 1000)")
    elementos_intervencao: List[ElementoIntervencao] = Field(
        default_factory=list,
        description="Elementos presentes na proposta de intervenção: agente, acao, modo_meio, finalidade, detalhamento",
    )
    pontos_fortes: List[str] = Field(min_length=1, description="De 1 a 4 pontos fortes concretos do texto")
    pontos_a_melhorar: List[str] = Field(min_length=1, description="De 1 a 4 melhorias concretas, com a competência envolvida")
    comentario_geral: Optional[str] = Field(None, description="Comentário final curto e encorajador")

    @field_validator("c1_norma_padrao", "c2_tema_repertorio", "c3_argumentacao",
                     "c4_coesao", "c5_intervencao", mode="before")
    @classmethod
    def ajustar_escala_enem(cls, v):
        """O ENEM só usa múltiplos de 40 entre 0 e 200. Se o modelo devolver
        150 ou '180', arredonda para o nível mais próximo (160)."""
        try:
            valor = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"nota de competência inválida: {v!r}")
        return min(NOTAS_VALIDAS, key=lambda nivel: abs(nivel - valor))

    @field_validator("elementos_intervencao", mode="before")
    @classmethod
    def normalizar_elementos(cls, v):
        """'Ação' → 'acao', 'modo/meio' → 'modo_meio'; descarta o que não for elemento."""
        validos = {"agente", "acao", "modo_meio", "finalidade", "detalhamento"}
        normalizados = [_slug(item) for item in (v or [])]
        normalizados = ["modo_meio" if x in ("modo", "meio", "modo_ou_meio") else x for x in normalizados]
        return list(dict.fromkeys(x for x in normalizados if x in validos))

    @field_validator("pontos_fortes", "pontos_a_melhorar", mode="before")
    @classmethod
    def limpar_listas(cls, v):
        return _limpar_lista(v)[:4]

    @model_validator(mode="after")
    def calcular_nota_total(self):
        """A nota total é SEMPRE a soma — não confiamos na conta do modelo."""
        self.nota_total = (self.c1_norma_padrao + self.c2_tema_repertorio + self.c3_argumentacao
                           + self.c4_coesao + self.c5_intervencao)
        return self


# ==============================================================
# 2) Relatório da sessão — qualquer matéria
# ==============================================================
class RelatorioSessao(BaseModel):
    materia: MateriaENEM = Field(description="Chave da matéria estudada")
    temas_estudados: List[str] = Field(min_length=1, description="Temas/conteúdos trabalhados na conversa")
    duvidas_principais: List[str] = Field(default_factory=list, description="Dúvidas ou erros que o aluno demonstrou")
    nivel_compreensao: Literal["iniciante", "intermediario", "avancado"] = Field(
        description="Nível de domínio que o aluno demonstrou na conversa"
    )
    proximos_passos: List[str] = Field(min_length=1, description="De 1 a 5 ações concretas de estudo")
    tempo_estudo_sugerido_min: int = Field(
        ge=10, le=180, description="Minutos sugeridos para a próxima sessão de estudo (10 a 180)"
    )
    observacao: Optional[str] = Field(None, description="Recado curto e motivador para o aluno")

    @field_validator("materia", mode="before")
    @classmethod
    def normalizar_materia(cls, v):
        """Aceita 'Matemática', 'História e Geografia', 'historia/geografia'…"""
        slug = _slug(v)
        if "historia" in slug or "geografia" in slug:
            return "historia_geografia"
        return slug

    @field_validator("nivel_compreensao", mode="before")
    @classmethod
    def normalizar_nivel(cls, v):
        return _slug(v)  # 'Intermediário' → 'intermediario'

    @field_validator("temas_estudados", "duvidas_principais", "proximos_passos", mode="before")
    @classmethod
    def limpar_listas(cls, v):
        return _limpar_lista(v)[:5]
