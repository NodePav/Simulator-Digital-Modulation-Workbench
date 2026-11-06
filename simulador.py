"""Simulador didatico de 12 tecnicas de modulacao digital.

Dependencias: numpy e matplotlib. Os sinais sao normalizados para energia
media por simbolo igual a 1; as metricas de banda sao estimativas didaticas.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


@dataclass(frozen=True)
class Configuracao:
    """Parametros globais usados na geracao dos sinais."""

    quantidade_bits: int = 1200
    taxa_bits: float = 1000.0
    frequencia_portadora: float = 5000.0
    roll_off: float = 0.35
    eb_n0_db: float = 10.0
    amostras_por_simbolo: int = 64
    semente: int = 42

    def validar(self) -> None:
        if self.quantidade_bits < 8:
            raise ValueError("Use pelo menos 8 bits para gerar os graficos.")
        if not math.isfinite(self.taxa_bits) or self.taxa_bits <= 0:
            raise ValueError("A taxa de bits Rb deve ser um valor positivo e finito.")
        if not math.isfinite(self.frequencia_portadora) or self.frequencia_portadora <= 0:
            raise ValueError("Rb e fc devem ser valores positivos.")
        if not math.isfinite(self.eb_n0_db):
            raise ValueError("Eb/N0 deve ser um valor finito.")
        if not math.isfinite(self.roll_off) or not 0 <= self.roll_off <= 1:
            raise ValueError("O fator roll-off alpha deve estar entre 0 e 1.")
        if self.amostras_por_simbolo < 8:
            raise ValueError("Use pelo menos 8 amostras por simbolo.")


@dataclass(frozen=True)
class Esquema:
    nome: str
    ordem: int
    tipo: str
    codificador: Callable[[np.ndarray], np.ndarray]

    @property
    def bits_por_simbolo(self) -> int:
        return int(np.log2(self.ordem))


@dataclass
class Resultado:
    esquema: Esquema
    bits: np.ndarray
    indices: np.ndarray
    simbolos: np.ndarray
    envelope: np.ndarray


def gray(indices: np.ndarray) -> np.ndarray:
    """Converte indices binarios em indices Gray."""
    return indices ^ (indices >> 1)


def codificar_ask(indices: np.ndarray) -> np.ndarray:
    # OOK com amplitudes 0 e sqrt(2), para energia media unitaria.
    return np.where(indices == 0, 0.0, np.sqrt(2.0)).astype(complex)


def codificar_psk(ordem: int) -> Callable[[np.ndarray], np.ndarray]:
    def codificar(indices: np.ndarray) -> np.ndarray:
        fases = 2 * np.pi * gray(indices) / ordem
        return np.exp(1j * fases)

    return codificar


def codificar_fsk(indices: np.ndarray) -> np.ndarray:
    # Representacao ortogonal equivalente: bit 0 no eixo I e bit 1 no eixo Q.
    return np.where(indices == 0, 1.0 + 0j, 0.0 + 1j)


def codificar_dpsk(indices: np.ndarray) -> np.ndarray:
    incrementos = np.where(indices == 0, 0.0, np.pi)
    return np.exp(1j * np.cumsum(incrementos))


def codificar_dqpsk(indices: np.ndarray) -> np.ndarray:
    # Incrementos Gray de 0, 90, 180 e 270 graus, acumulados simbolo a simbolo.
    tabela_fases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    return np.exp(1j * np.cumsum(tabela_fases[gray(indices)]))


def codificar_pi4_dqpsk(indices: np.ndarray) -> np.ndarray:
    # As transicoes deixam a constelacao alternar entre dois conjuntos pi/4.
    tabela_fases = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    return np.exp(1j * np.cumsum(tabela_fases[gray(indices)]))


def codificar_oqpsk(indices: np.ndarray) -> np.ndarray:
    # A forma de onda aplica o atraso de meio simbolo ao ramo Q.
    return codificar_psk(4)(indices)


def codificar_qam16(indices: np.ndarray) -> np.ndarray:
    # Cada eixo usa codificacao Gray: 00, 01, 10, 11 -> -3, -1, +3, +1.
    niveis_gray = np.array([-3.0, -1.0, 3.0, 1.0])
    componente_i = niveis_gray[(indices >> 2) & 0b11]
    componente_q = niveis_gray[indices & 0b11]
    return (componente_i + 1j * componente_q) / np.sqrt(10.0)


def codificar_qam_diferencial(indices: np.ndarray) -> np.ndarray:
    # Variante didatica: amplitude QAM instantanea e fase acumulada.
    pontos = codificar_qam16(indices)
    fases_diferenciais = np.angle(pontos)
    amplitudes = np.abs(pontos)
    return amplitudes * np.exp(1j * np.cumsum(fases_diferenciais))


ESQUEMAS = (
    Esquema("ASK / OOK", 2, "linear", codificar_ask),
    Esquema("BFSK", 2, "frequencia", codificar_fsk),
    Esquema("M-PSK (8-PSK)", 8, "fase", codificar_psk(8)),
    Esquema("BPSK", 2, "fase", codificar_psk(2)),
    Esquema("QPSK", 4, "fase", codificar_psk(4)),
    Esquema("DPSK", 2, "diferencial", codificar_dpsk),
    Esquema("DQPSK", 4, "diferencial", codificar_dqpsk),
    Esquema("OQPSK", 4, "fase", codificar_oqpsk),
    Esquema("pi/4-DQPSK", 4, "diferencial", codificar_pi4_dqpsk),
    Esquema("MSK (h=0.5)", 2, "frequencia", codificar_fsk),
    Esquema("16-QAM", 16, "linear", codificar_qam16),
    Esquema("16-QAM diferencial", 16, "diferencial", codificar_qam_diferencial),
)

DESCRICOES_ESQUEMAS = {
    "ASK / OOK": (
        "Significado: representa os bits variando a amplitude da portadora; "
        "no OOK, um dos bits desliga a portadora.\n"
        "Aplicações comuns: enlaces ópticos simples, RFID de baixo custo e "
        "controles remotos por infravermelho."
    ),
    "BFSK": (
        "Significado: codifica 0 e 1 usando duas frequências (tons) distintas.\n"
        "Aplicações comuns: telemetria, modems simples, identificação por rádio "
        "e comunicação robusta de baixa velocidade."
    ),
    "M-PSK (8-PSK)": (
        "Significado: usa oito fases possíveis; cada símbolo transporta 3 bits.\n"
        "Aplicações comuns: enlaces digitais com boa eficiência espectral e "
        "sistemas de satélite, quando a relação sinal-ruído é suficiente."
    ),
    "BPSK": (
        "Significado: usa duas fases separadas por 180° para representar os bits.\n"
        "Aplicações comuns: telemetria, enlaces de espaço profundo e canais "
        "com ruído, devido à boa robustez."
    ),
    "QPSK": (
        "Significado: usa quatro fases; cada símbolo representa 2 bits.\n"
        "Aplicações comuns: comunicações celulares e via satélite, Wi-Fi e "
        "enlaces digitais que equilibram eficiência e robustez."
    ),
    "DPSK": (
        "Significado: o bit é codificado na mudança de fase entre símbolos "
        "consecutivos, sem exigir uma fase de referência absoluta.\n"
        "Aplicações comuns: enlaces simples ou sujeitos a variações de fase "
        "onde se deseja simplificar a demodulação."
    ),
    "DQPSK": (
        "Significado: variante diferencial de QPSK; os 2 bits indicam a "
        "mudança de fase em relação ao símbolo anterior.\n"
        "Aplicações comuns: enlaces de rádio e sistemas que simplificam a "
        "recuperação da referência de fase."
    ),
    "OQPSK": (
        "Significado: QPSK com os ramos I e Q deslocados por meio símbolo, "
        "evitando mudanças simultâneas grandes de fase.\n"
        "Aplicações comuns: rádio via satélite e enlaces com amplificadores "
        "não lineares ou de alta eficiência."
    ),
    "pi/4-DQPSK": (
        "Significado: modulação diferencial de quatro fases com duas "
        "constelações alternadas, deslocadas em π/4.\n"
        "Aplicações comuns: sistemas de rádio móvel, incluindo padrões "
        "legados como PHS e variantes de equipamentos TETRA."
    ),
    "MSK (h=0.5)": (
        "Significado: FSK de fase contínua com índice h=0,5; mantém o envelope "
        "constante e transições suaves.\n"
        "Aplicações comuns: telemetria e enlaces de rádio eficientes; formas "
        "relacionadas incluem GMSK em GSM."
    ),
    "16-QAM": (
        "Significado: combina 4 níveis em I e 4 em Q, formando 16 estados "
        "de amplitude e fase; transporta 4 bits por símbolo.\n"
        "Aplicações comuns: Wi-Fi, redes celulares, TV digital e modems "
        "de banda larga."
    ),
    "16-QAM diferencial": (
        "Significado: demonstração didática que combina os pontos de 16-QAM "
        "com acumulação diferencial de fase.\n"
        "Aplicações comuns: conceito útil para estudar codificação diferencial; "
        "esta implementação não representa um padrão DQAM universal."
    ),
}


class SimuladorModulacao:
    """Gera formas de onda, constelacoes e um painel de comparacao."""

    def __init__(self, configuracao: Configuracao) -> None:
        configuracao.validar()
        self.config = configuracao
        self.rng = np.random.default_rng(configuracao.semente)
        self.bits = self.rng.integers(0, 2, configuracao.quantidade_bits, dtype=np.int8)

    def _agrupar_bits(self, bits_por_simbolo: int) -> tuple[np.ndarray, np.ndarray]:
        bits = self.bits.copy()
        resto = len(bits) % bits_por_simbolo
        if resto:
            bits = np.pad(bits, (0, bits_por_simbolo - resto))
        grupos = bits.reshape(-1, bits_por_simbolo)
        pesos = 1 << np.arange(bits_por_simbolo - 1, -1, -1)
        return bits, grupos @ pesos

    def _envolvente(
        self, esquema: Esquema, simbolos: np.ndarray
    ) -> np.ndarray:
        sps = self.config.amostras_por_simbolo
        rs = self.config.taxa_bits / esquema.bits_por_simbolo

        if esquema.nome == "BFSK":
            # BFSK nao coerente: cada tom inicia com fase zero.
            desvio = rs / 4
            frequencias = np.repeat(
                np.where(np.imag(simbolos) > 0, desvio, -desvio), sps
            )
            tempos_locais = np.tile(np.arange(sps) / (sps * rs), len(simbolos))
            return np.exp(1j * 2 * np.pi * frequencias * tempos_locais)

        if esquema.nome == "MSK (h=0.5)":
            # CPFSK com h=0.5: a fase integra o desvio e permanece continua.
            desvio = 0.25 * rs
            frequencias = np.repeat(
                np.where(np.imag(simbolos) > 0, desvio, -desvio), sps
            )
            fase = 2 * np.pi * np.cumsum(frequencias) / (sps * rs)
            return np.exp(1j * fase)

        if esquema.nome == "OQPSK":
            ramo_i = np.repeat(np.real(simbolos), sps)
            ramo_q = np.repeat(np.imag(simbolos), sps)
            atraso = sps // 2
            ramo_q_atrasado = np.concatenate(
                (np.zeros(atraso), ramo_q[:-atraso])
            )
            return ramo_i[: len(ramo_q_atrasado)] + 1j * ramo_q_atrasado

        return np.repeat(simbolos, sps)

    def gerar_resultado(self, esquema: Esquema) -> Resultado:
        bits, indices = self._agrupar_bits(esquema.bits_por_simbolo)
        simbolos = esquema.codificador(indices)
        simbolos = simbolos / np.sqrt(np.mean(np.abs(simbolos) ** 2))
        envelope = self._envolvente(esquema, simbolos)
        return Resultado(esquema, bits, indices, simbolos, envelope)

    def metricas(self, esquema: Esquema, resultado: Resultado) -> dict[str, float]:
        k = esquema.bits_por_simbolo
        rb = self.config.taxa_bits
        rs = rb / k
        energia_simbolo = float(np.mean(np.abs(resultado.simbolos) ** 2))
        energia_bit = energia_simbolo / k
        potencia_media = float(np.mean(np.abs(resultado.envelope) ** 2))
        pico = float(np.max(np.abs(resultado.envelope) ** 2))
        return {
            "M": esquema.ordem,
            "k": k,
            "Tb": 1 / rb,
            "Ts": 1 / rs,
            "Rs": rs,
            "B": (1 + self.config.roll_off) * rs,
            "Es": energia_simbolo,
            "Eb": energia_bit,
            "eficiencia": k / (1 + self.config.roll_off),
            "papr": pico / potencia_media if potencia_media else 0.0,
        }

    def adicionar_ruido(self, envelope: np.ndarray, k: int) -> np.ndarray:
        """Adiciona AWGN complexo com variancia definida pelo Eb/N0 configurado."""
        eb_n0 = 10 ** (self.config.eb_n0_db / 10)
        variancia_por_componente = 1 / (2 * k * eb_n0)
        ruido = self.rng.normal(0, np.sqrt(variancia_por_componente), len(envelope))
        ruido = ruido + 1j * self.rng.normal(
            0, np.sqrt(variancia_por_componente), len(envelope)
        )
        return envelope + ruido

    def plot_forma_de_onda(self, resultado: Resultado) -> None:
        sps = self.config.amostras_por_simbolo
        k = resultado.esquema.bits_por_simbolo
        rs = self.config.taxa_bits / k
        if self.config.frequencia_portadora + rs / 4 >= sps * rs / 2:
            raise ValueError(
                "fc esta acima do limite de Nyquist para as amostras por simbolo."
            )
        amostras = min(len(resultado.envelope), 12 * sps)
        tempo = np.arange(amostras) / (sps * rs)
        portadora = np.exp(
            1j * 2 * np.pi * self.config.frequencia_portadora * tempo
        )
        sinal = np.real(resultado.envelope[:amostras] * portadora)
        recebido = np.real(self.adicionar_ruido(resultado.envelope[:amostras], k) * portadora)

        bits_mostrar = resultado.bits[: min(len(resultado.bits), 12 * k)]
        tempo_bits = np.arange(len(bits_mostrar) + 1) / self.config.taxa_bits
        nivel_bits = np.append(bits_mostrar, bits_mostrar[-1])

        fig, eixos = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        eixos[0].step(tempo_bits, nivel_bits, where="post", color="black")
        eixos[0].set_ylabel("Bit")
        eixos[0].set_ylim(-0.2, 1.2)
        eixos[0].grid(True, alpha=0.3)
        eixos[1].plot(tempo, sinal, label="Sinal modulado", linewidth=1)
        eixos[1].plot(tempo, recebido, label="Com AWGN", alpha=0.55, linewidth=0.8)
        eixos[1].set_xlabel("Tempo (s)")
        eixos[1].set_ylabel("Amplitude")
        eixos[1].grid(True, alpha=0.3)
        eixos[1].legend()
        fig.suptitle(f"Forma de onda: {resultado.esquema.nome}")
        fig.tight_layout()

    def plot_constelacoes(self) -> None:
        fig, eixos = plt.subplots(3, 4, figsize=(13, 9))
        for eixo, esquema in zip(eixos.flat, ESQUEMAS):
            resultado = self.gerar_resultado(esquema)
            eixo.scatter(
                resultado.simbolos.real,
                resultado.simbolos.imag,
                s=12,
                alpha=0.45,
                label="Simbolos gerados",
            )
            eixo.axhline(0, color="gray", linewidth=0.6)
            eixo.axvline(0, color="gray", linewidth=0.6)
            eixo.set_title(esquema.nome, fontsize=9)
            eixo.set_xlabel("I")
            eixo.set_ylabel("Q")
            eixo.grid(True, alpha=0.25)
            eixo.set_aspect("equal", adjustable="datalim")
        fig.suptitle("Constelacoes I/Q — simbolos mapeados")
        fig.tight_layout()

    def plot_dashboard(self) -> None:
        dados = [
            (esquema, self.metricas(esquema, self.gerar_resultado(esquema)))
            for esquema in ESQUEMAS
        ]
        nomes = [esquema.nome for esquema, _ in dados]
        x = np.arange(len(nomes))
        largura = 0.72
        fig, eixos = plt.subplots(3, 1, figsize=(13, 10))
        itens = (
            ("Rs", "Taxa de simbolos (simbolos/s)"),
            ("B", "Banda estimada (Hz)"),
            ("eficiencia", "Eficiencia espectral estimada (bit/s/Hz)"),
        )
        for eixo, (chave, titulo) in zip(eixos, itens):
            valores = [metricas[chave] for _, metricas in dados]
            eixo.bar(x, valores, width=largura, color="#4472C4")
            eixo.set_ylabel(titulo)
            eixo.grid(axis="y", alpha=0.3)
            eixo.set_xticks(x, nomes, rotation=25, ha="right")
        fig.suptitle(
            f"Comparacao didatica | Rb={self.config.taxa_bits:g} bit/s, "
            f"alpha={self.config.roll_off:g}"
        )
        fig.tight_layout()

        print("\nMetricas (energia media por simbolo normalizada):")
        print(
            f"{'Tecnica':<23} {'M':>3} {'k':>3} {'Tb (s)':>10} {'Ts (s)':>10} "
            f"{'Rs (Bd)':>12} {'B aprox. (Hz)':>15} {'Eb':>9} {'Es':>9} {'PAPR':>8}"
        )
        for esquema, metricas in dados:
            print(
                f"{esquema.nome:<23} {metricas['M']:>3.0f} {metricas['k']:>3.0f} "
                f"{metricas['Tb']:>10.6f} {metricas['Ts']:>10.6f} "
                f"{metricas['Rs']:>12.2f} {metricas['B']:>15.2f} "
                f"{metricas['Eb']:>9.4f} {metricas['Es']:>9.4f} "
                f"{metricas['papr']:>8.3f}"
            )

    def executar(self, nome_esquema: str) -> None:
        esquema = next(item for item in ESQUEMAS if item.nome == nome_esquema)
        resultado = self.gerar_resultado(esquema)
        print(
            f"Bits gerados: {len(self.bits)} | Tecnica exibida: {esquema.nome} | "
            f"Eb/N0: {self.config.eb_n0_db:g} dB"
        )
        self.plot_forma_de_onda(resultado)
        self.plot_constelacoes()
        self.plot_dashboard()
        plt.show()


class JanelaSimulador:
    """Interface grafica Tkinter para configurar e visualizar as modulacoes."""

    CORES = {
        "fundo": "#eef2f6",
        "painel": "#ffffff",
        "marinho": "#10263f",
        "marinho_claro": "#193b5a",
        "ciano": "#16c5d8",
        "texto": "#20364b",
        "grade": "#dce5ed",
        "azul": "#2878b5",
        "verde": "#20a486",
        "ambar": "#e9a23b",
    }

    def __init__(self, raiz: tk.Tk, configuracao_inicial: Configuracao, esquema_inicial: str) -> None:
        self.raiz = raiz
        self.raiz.title("MODLAB | Digital Modulation Workbench")
        self.raiz.geometry("1440x920")
        self.raiz.minsize(1100, 740)
        self.raiz.configure(background=self.CORES["fundo"])

        self.variaveis = {
            "bits": tk.StringVar(value=str(configuracao_inicial.quantidade_bits)),
            "rb": tk.StringVar(value=str(configuracao_inicial.taxa_bits)),
            "fc": tk.StringVar(value=str(configuracao_inicial.frequencia_portadora)),
            "alpha": tk.StringVar(value=str(configuracao_inicial.roll_off)),
            "ebn0": tk.StringVar(value=str(configuracao_inicial.eb_n0_db)),
        }
        self.configuracao = configuracao_inicial
        self.simulador: SimuladorModulacao | None = None
        self._agendador_atualizacao: str | None = None
        self._montar_interface()
        for variavel in self.variaveis.values():
            variavel.trace_add("write", self._agendar_atualizacao)
        self.lista_esquemas.selection_set(
            next(
                (indice for indice, item in enumerate(ESQUEMAS) if item.nome == esquema_inicial),
                4,
            )
        )
        self._atualizar_descricao_tecnica()
        self.atualizar_simulacao()

    def _montar_interface(self) -> None:
        estilo = ttk.Style(self.raiz)
        estilo.theme_use("clam")
        estilo.configure("TFrame", background=self.CORES["fundo"])
        estilo.configure("Card.TFrame", background=self.CORES["painel"])
        estilo.configure(
            "TLabel", background=self.CORES["fundo"],
            foreground=self.CORES["texto"], font=("Segoe UI", 10),
        )
        estilo.configure(
            "Title.TLabel", font=("Segoe UI", 17, "bold"),
            foreground=self.CORES["marinho"],
        )
        estilo.configure(
            "Accent.TButton", font=("Segoe UI", 10, "bold"), padding=9,
            background=self.CORES["ciano"], foreground=self.CORES["marinho"],
        )
        estilo.map("Accent.TButton", background=[("active", "#55d9e5")])
        estilo.configure("TNotebook", background=self.CORES["fundo"], borderwidth=0)
        estilo.configure(
            "TNotebook.Tab", padding=(15, 9), font=("Segoe UI", 10, "bold"),
            background="#dfe7ee", foreground=self.CORES["texto"],
        )
        estilo.map(
            "TNotebook.Tab",
            background=[("selected", self.CORES["painel"])],
            foreground=[("selected", self.CORES["azul"])],
        )

        principal = ttk.Frame(self.raiz, padding=(18, 16, 18, 14))
        principal.pack(fill=tk.BOTH, expand=True)
        cabecalho = tk.Frame(principal, bg=self.CORES["marinho"], height=100)
        cabecalho.pack(fill=tk.X, pady=(0, 14))
        cabecalho.pack_propagate(False)
        marca = tk.Canvas(
            cabecalho, width=58, height=58, bg=self.CORES["marinho"],
            highlightthickness=0,
        )
        marca.pack(side=tk.LEFT, padx=(18, 10), pady=20)
        marca.create_oval(5, 5, 53, 53, outline=self.CORES["ciano"], width=2)
        marca.create_arc(
            13, 13, 45, 45, start=35, extent=290,
            outline="#ffffff", style=tk.ARC, width=2,
        )
        marca.create_line(16, 31, 24, 31, fill=self.CORES["ciano"], width=2)
        marca.create_line(24, 31, 30, 20, fill=self.CORES["ciano"], width=2)
        marca.create_line(30, 20, 36, 41, fill=self.CORES["ciano"], width=2)
        marca.create_line(36, 41, 43, 26, fill=self.CORES["ciano"], width=2)
        identidade = tk.Frame(cabecalho, bg=self.CORES["marinho"])
        identidade.pack(side=tk.LEFT, fill=tk.Y, pady=10)
        tk.Label(
            identidade, text="MODLAB", bg=self.CORES["marinho"], fg="#ffffff",
            font=("Segoe UI", 19, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            identidade, text="DIGITAL MODULATION WORKBENCH",
            bg=self.CORES["marinho"], fg="#a9c1d5",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            identidade,
            text="PAVÃO ENGENHARIA  •  PROPRIETÁRIO E CRIADOR: JOSUÉ PAVÃO",
            bg=self.CORES["marinho"], fg=self.CORES["ciano"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor=tk.W, pady=(3, 0))
        tk.Label(
            cabecalho, text="SIGNAL PROCESSING  /  LAB 01",
            bg=self.CORES["marinho"], fg=self.CORES["ciano"],
            font=("Consolas", 10, "bold"),
        ).pack(side=tk.RIGHT, padx=20)
        ttk.Label(
            principal,
            text="Estação de análise de modulações digitais",
            style="Title.TLabel",
        ).pack(anchor=tk.W, pady=(0, 12))

        corpo = ttk.Frame(principal)
        corpo.pack(fill=tk.BOTH, expand=True)
        barra_lateral = tk.Frame(
            corpo, bg=self.CORES["marinho"], padx=14, pady=14, width=260,
        )
        barra_lateral.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        barra_lateral.pack_propagate(False)
        tk.Label(
            barra_lateral, text="CONFIGURAÇÃO", bg=self.CORES["marinho"],
            fg=self.CORES["ciano"], font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(2, 10))
        tk.Label(
            barra_lateral, text="Técnica de modulação",
            bg=self.CORES["marinho"], fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))
        self.lista_esquemas = tk.Listbox(
            barra_lateral,
            height=len(ESQUEMAS),
            width=27,
            exportselection=False,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            highlightthickness=0,
            background=self.CORES["marinho_claro"],
            foreground="#e5eef5",
            selectbackground=self.CORES["ciano"],
            selectforeground="white",
        )
        for esquema in ESQUEMAS:
            self.lista_esquemas.insert(tk.END, esquema.nome)
        self.lista_esquemas.pack(fill=tk.X)
        self.lista_esquemas.bind("<<ListboxSelect>>", self._ao_selecionar_esquema)

        tk.Label(
            barra_lateral, text="SOBRE A TÉCNICA",
            bg=self.CORES["marinho"], fg=self.CORES["ciano"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor=tk.W, pady=(10, 4))
        self.descricao_tecnica = tk.Label(
            barra_lateral,
            text="",
            bg=self.CORES["marinho_claro"],
            fg="#e5eef5",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            anchor=tk.NW,
            wraplength=220,
            padx=9,
            pady=8,
        )
        self.descricao_tecnica.pack(fill=tk.X)

        tk.Frame(barra_lateral, bg="#36516a", height=1).pack(fill=tk.X, pady=12)
        tk.Label(
            barra_lateral, text="PARÂMETROS DO SINAL",
            bg=self.CORES["marinho"], fg=self.CORES["ciano"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 4))
        tk.Label(
            barra_lateral, text="Os resultados acompanham as entradas.",
            bg=self.CORES["marinho"], fg="#a9c1d5", font=("Segoe UI", 8),
            wraplength=220, justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 5))
        campos = (
            ("BITS DA SEQUÊNCIA", "bits"),
            ("TAXA DE BITS  Rb (bit/s)", "rb"),
            ("PORTADORA  fc (Hz)", "fc"),
            ("ROLL-OFF  α (0–1)", "alpha"),
            ("Eb/N₀ (dB)", "ebn0"),
        )
        for rotulo, chave in campos:
            tk.Label(
                barra_lateral, text=rotulo, bg=self.CORES["marinho"],
                fg="#c5d4e0", font=("Segoe UI", 8, "bold"),
            ).pack(anchor=tk.W, pady=(7, 3))
            tk.Entry(
                barra_lateral, textvariable=self.variaveis[chave],
                font=("Consolas", 10), relief=tk.FLAT, bd=0,
                background="#f3f7fa", foreground=self.CORES["marinho"],
                insertbackground=self.CORES["azul"],
            ).pack(fill=tk.X, ipady=6)
        ttk.Button(
            barra_lateral, text="⟳  Atualizar agora",
            command=self.atualizar_simulacao, style="Accent.TButton",
        ).pack(fill=tk.X, pady=(13, 4))
        tk.Label(
            barra_lateral, text="AUTO-UPDATE  •  400 ms",
            bg=self.CORES["marinho"], fg="#83a0b7", font=("Consolas", 8),
        ).pack(anchor=tk.CENTER, pady=(5, 0))

        area = ttk.Frame(corpo)
        area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            area, text="  LIVE TELEMETRY  /  MEMORIAL DE CÁLCULO",
            bg=self.CORES["marinho_claro"], fg="#ffffff",
            font=("Segoe UI", 9, "bold"), anchor=tk.W, padx=10, pady=6,
        ).pack(fill=tk.X)
        self.texto_metricas = tk.Text(
            area, height=7, wrap=tk.WORD, font=("Consolas", 10),
            relief=tk.FLAT, padx=12, pady=9, background=self.CORES["painel"],
            foreground=self.CORES["texto"], highlightthickness=1,
            highlightbackground=self.CORES["grade"],
        )
        self.texto_metricas.pack(fill=tk.X, pady=(0, 7))
        self.status_var = tk.StringVar(value="●  SISTEMA PRONTO")
        self.status_label = tk.Label(
            area, textvariable=self.status_var, anchor=tk.W,
            bg=self.CORES["fundo"], fg=self.CORES["verde"],
            font=("Consolas", 9, "bold"),
        )
        self.status_label.pack(fill=tk.X, pady=(0, 6))
        self.abas = ttk.Notebook(area)
        self.abas.pack(fill=tk.BOTH, expand=True)
        self.figuras: dict[str, Figure] = {}
        self.eixos: dict[str, tuple] = {}
        self.canvases: dict[str, FigureCanvasTkAgg] = {}
        for chave, rotulo, tamanho, numero_eixos in (
            ("onda", "◉  Forma de onda", (10, 5.5), 2),
            ("constelacao", "⌖  Constelação I/Q", (8, 6), 1),
            ("dashboard", "▥  Comparativo", (11, 6), 2),
        ):
            aba = ttk.Frame(self.abas, padding=8)
            self.abas.add(aba, text=rotulo)
            figura = Figure(
                figsize=tamanho, dpi=100, constrained_layout=True,
                facecolor=self.CORES["painel"],
            )
            if numero_eixos == 1:
                eixos = (figura.add_subplot(111),)
            else:
                eixos = tuple(figura.subplots(numero_eixos, 1, sharex=False))
            for eixo in eixos:
                eixo.set_facecolor(self.CORES["painel"])
            canvas = FigureCanvasTkAgg(figura, master=aba)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figuras[chave] = figura
            self.eixos[chave] = eixos
            self.canvases[chave] = canvas

        aba_ajuda = ttk.Frame(self.abas, padding=12)
        self.abas.add(aba_ajuda, text="?  Ajuda")
        self._montar_ajuda(aba_ajuda)

    def _montar_ajuda(self, aba: ttk.Frame) -> None:
        """Cria um guia de uso embutido, com rolagem para janelas menores."""
        moldura = ttk.Frame(aba)
        moldura.pack(fill=tk.BOTH, expand=True)
        texto = tk.Text(
            moldura, wrap=tk.WORD, font=("Segoe UI", 10),
            background=self.CORES["painel"], foreground=self.CORES["texto"],
            relief=tk.FLAT, padx=24, pady=20, spacing1=3, spacing3=5,
        )
        rolagem = ttk.Scrollbar(moldura, orient=tk.VERTICAL, command=texto.yview)
        texto.configure(yscrollcommand=rolagem.set)
        rolagem.pack(side=tk.RIGHT, fill=tk.Y)
        texto.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        texto.tag_configure(
            "titulo", font=("Segoe UI", 18, "bold"),
            foreground=self.CORES["marinho"], spacing3=8,
        )
        texto.tag_configure(
            "secao", font=("Segoe UI", 12, "bold"),
            foreground=self.CORES["azul"], spacing1=13, spacing3=5,
        )
        texto.tag_configure(
            "passo", font=("Segoe UI", 10, "bold"),
            foreground=self.CORES["marinho"], lmargin1=8,
        )
        texto.tag_configure(
            "corpo", font=("Segoe UI", 10), foreground=self.CORES["texto"],
            lmargin1=8, lmargin2=8, spacing3=7,
        )
        texto.tag_configure(
            "formula", font=("Consolas", 10), foreground=self.CORES["verde"],
            lmargin1=18, spacing3=7,
        )
        texto.insert(tk.END, "GUIA RÁPIDO  /  MODLAB\n", "titulo")
        texto.insert(
            tk.END,
            "Simule, compare e interprete técnicas de modulação digital. "
            "Os resultados e gráficos são atualizados automaticamente.\n",
            "corpo",
        )
        secoes = (
            (
                "01  |  PRIMEIROS PASSOS",
                (
                    ("1. Escolha a modulação", "Selecione uma técnica na lista à esquerda. "
                     "A seleção atualiza a forma de onda, a constelação e os dados."),
                    ("2. Defina a quantidade de bits", "Digite um inteiro maior ou igual a 8. "
                     "A sequência pseudoaleatória será gerada com esse tamanho."),
                    ("3. Ajuste os parâmetros", "Informe Rb em bit/s, fc em Hz, α entre 0 e 1 "
                     "e Eb/N₀ em dB. Os dados atualizam após uma breve pausa."),
                    ("4. Explore os resultados", "Abra as abas Forma de onda, Constelação I/Q, "
                     "Comparativo e Ajuda clicando em seus títulos."),
                    ("5. Recálculo manual", "Use Atualizar agora para atualizar imediatamente. "
                     "Entradas inválidas são descritas na barra de estado."),
                ),
            ),
            (
                "02  |  COMO LER AS VISUALIZAÇÕES",
                (
                    ("Forma de onda — eixos e linhas", "O eixo horizontal é o tempo em segundos. "
                     "O primeiro gráfico mostra os bits no tempo (0 ou 1). No segundo, o eixo "
                     "vertical é a amplitude instantânea: azul = sinal modulado ideal; laranja "
                     "= sinal com ruído AWGN. A legenda laranja informa o Eb/N₀ aplicado. "
                     "São exibidos até 12 períodos de símbolo."),
                    ("Constelação I/Q — coordenadas", "O eixo horizontal mostra I (componente "
                     "em fase) e o vertical mostra Q (quadratura), em amplitudes normalizadas. "
                     "Pontos azuis são símbolos transmitidos; círculos vazados destacam estados "
                     "distintos observados. Repetições ficam sobrepostas."),
                    ("Exceções da constelação", "Na BFSK, I/Q representa dois sinais ortogonais "
                     "(os tons), não a constelação usual de PSK. Em técnicas diferenciais, "
                     "a informação está na transição entre símbolos; os estados dependem do "
                     "histórico da sequência."),
                    ("Dashboard — taxa de símbolos", "A primeira barra mostra Rs = Rb/k, em "
                     "símbolos por segundo (baud). Os valores numéricos aparecem acima de cada "
                     "barra. Com Rb fixo, aumentar k reduz Rs."),
                    ("Dashboard — eficiência espectral", "A segunda barra mostra a estimativa "
                     "η ≈ k/(1+α), em bit/s/Hz. Barras maiores indicam mais bits por segundo "
                     "por hertz segundo essa aproximação, não uma medida de desempenho real."),
                ),
            ),
            (
                "03  |  MEMORIAL DE CÁLCULO",
                (
                    ("M e k", "M é o número de estados e k = log₂(M) informa quantos bits "
                     "são representados por símbolo."),
                    ("Taxas e períodos", "Rs = Rb/k; Tb = 1/Rb é o período de bit; "
                     "Ts = 1/Rs é o período de símbolo."),
                    ("Banda e energia", "B ≈ (1+α)Rs é uma estimativa didática. "
                     "Os símbolos são normalizados para Es = 1 e Eb = Es/k."),
                ),
            ),
            (
                "04  |  DICAS E LIMITAÇÕES",
                (
                    ("Valores de entrada", "Rb e fc devem ser positivos; α deve ficar "
                     "entre 0 e 1. Quantidade de bits é um inteiro >= 8."),
                    ("Forma de onda", "Se fc estiver acima do limite representável pela "
                     "resolução temporal, reduza fc ou aumente Rb."),
                    ("Modelo didático", "Os gráficos usam pulsos retangulares. A banda de "
                     "FSK/MSK depende dos tons e do critério espectral; a fórmula exibida "
                     "é uma aproximação."),
                ),
            ),
        )
        for titulo_secao, itens in secoes:
            texto.insert(tk.END, f"\n{titulo_secao}\n", "secao")
            for passo, explicacao in itens:
                texto.insert(tk.END, f"{passo}\n", "passo")
                texto.insert(tk.END, f"{explicacao}\n", "corpo")
        texto.insert(tk.END, "\nFÓRMULAS DE REFERÊNCIA\n", "secao")
        texto.insert(
            tk.END, "k = log₂(M)     Rs = Rb/k     Tb = 1/Rb     Ts = 1/Rs\n",
            "formula",
        )
        texto.insert(
            tk.END, "B ≈ (1+α)Rs     Es = 1 (normalizado)     Eb = Es/k\n",
            "formula",
        )
        texto.configure(state=tk.DISABLED)

    def _ao_selecionar_esquema(self, evento: tk.Event) -> None:
        if evento.widget is self.lista_esquemas and self.lista_esquemas.curselection():
            self._atualizar_descricao_tecnica()
            self._agendar_atualizacao()

    def _atualizar_descricao_tecnica(self) -> None:
        esquema = self._esquema_atual()
        self.descricao_tecnica.configure(
            text=f"{esquema.nome}\n\n{DESCRICOES_ESQUEMAS[esquema.nome]}"
        )

    def _agendar_atualizacao(self, *trace_info: str) -> None:
        if trace_info and trace_info[-1] != "write":
            return
        if self._agendador_atualizacao is not None:
            self.raiz.after_cancel(self._agendador_atualizacao)
        self.status_var.set("◌  PARÂMETROS ALTERADOS — ATUALIZANDO")
        self.status_label.configure(fg=self.CORES["ambar"])
        self._agendador_atualizacao = self.raiz.after(
            400, self.atualizar_simulacao
        )

    def _ler_configuracao(self) -> Configuracao:
        try:
            quantidade_bits = int(self.variaveis["bits"].get())
            taxa_bits = float(self.variaveis["rb"].get())
            frequencia_portadora = float(self.variaveis["fc"].get())
            roll_off = float(self.variaveis["alpha"].get())
            eb_n0_db = float(self.variaveis["ebn0"].get())
        except ValueError as erro:
            raise ValueError(
                "Verifique os campos: quantidade de bits deve ser inteiro e os "
                "demais parâmetros devem ser numéricos."
            ) from erro
        return Configuracao(
            quantidade_bits=quantidade_bits,
            taxa_bits=taxa_bits,
            frequencia_portadora=frequencia_portadora,
            roll_off=roll_off,
            eb_n0_db=eb_n0_db,
            semente=self.configuracao.semente,
        )

    def atualizar_simulacao(self) -> None:
        if self._agendador_atualizacao is not None:
            self.raiz.after_cancel(self._agendador_atualizacao)
            self._agendador_atualizacao = None
        try:
            self.configuracao = self._ler_configuracao()
            self.configuracao.validar()
            self.simulador = SimuladorModulacao(self.configuracao)
            self._desenhar_resultados()
            self.status_var.set("●  SIMULAÇÃO ATUALIZADA")
            self.status_label.configure(fg=self.CORES["verde"])
        except ValueError as erro:
            self.status_var.set(f"!  PARÂMETROS INVÁLIDOS — {erro}")
            self.status_label.configure(fg="#bd4b43")

    def _esquema_atual(self) -> Esquema:
        selecao = self.lista_esquemas.curselection()
        indice = selecao[0] if selecao else 4
        return ESQUEMAS[indice]

    def _desenhar_resultados(self) -> None:
        if self.simulador is None:
            return
        esquema = self._esquema_atual()
        resultado = self.simulador.gerar_resultado(esquema)
        metricas = self.simulador.metricas(esquema, resultado)
        self._desenhar_metricas(esquema, metricas)
        self._desenhar_forma_de_onda(resultado)
        self._desenhar_constelacao(resultado)
        self._desenhar_dashboard()

    def _desenhar_metricas(self, esquema: Esquema, metricas: dict[str, float]) -> None:
        conteudo = (
            f"Técnica selecionada: {esquema.nome}   |   "
            f"Bits gerados: {len(self.simulador.bits) if self.simulador else 0}\n"
            f"M = {metricas['M']} estados   |   k = log₂(M) = {metricas['k']} bit/símbolo\n"
            f"Rb = {self.configuracao.taxa_bits:g} bit/s   |   "
            f"Rs = Rb/k = {metricas['Rs']:.3f} baud\n"
            f"Tb = {metricas['Tb']:.6g} s   |   Ts = 1/Rs = {metricas['Ts']:.6g} s\n"
            f"B ≈ (1+α)Rs = {metricas['B']:.3f} Hz   |   "
            f"α = {self.configuracao.roll_off:g}\n"
            f"Es = {metricas['Es']:.4f}   |   Eb = Es/k = {metricas['Eb']:.4f}   |   "
            f"Eb/N₀ = {self.configuracao.eb_n0_db:g} dB"
        )
        self.texto_metricas.configure(state=tk.NORMAL)
        self.texto_metricas.delete("1.0", tk.END)
        self.texto_metricas.insert("1.0", conteudo)
        self.texto_metricas.configure(state=tk.DISABLED)

    def _desenhar_forma_de_onda(self, resultado: Resultado) -> None:
        eixo_bits, eixo_sinal = self.eixos["onda"]
        eixo_bits.clear()
        eixo_sinal.clear()
        configuracao = self.configuracao
        sps = configuracao.amostras_por_simbolo
        k = resultado.esquema.bits_por_simbolo
        rs = configuracao.taxa_bits / k
        if configuracao.frequencia_portadora + rs / 4 >= sps * rs / 2:
            raise ValueError(
                "fc excede Nyquist para a resolução atual. Reduza fc ou Rb, "
                "ou aumente amostras_por_simbolo no código."
            )
        quantidade_simbolos = min(len(resultado.simbolos), 12)
        quantidade_amostras = quantidade_simbolos * sps
        tempo = np.arange(quantidade_amostras) / (sps * rs)
        bits_exibidos = resultado.bits[: quantidade_simbolos * k]
        tempos_bits = np.arange(len(bits_exibidos) + 1) / configuracao.taxa_bits
        niveis_bits = np.append(bits_exibidos, bits_exibidos[-1])
        eixo_bits.step(
            tempos_bits, niveis_bits, where="post",
            color=self.CORES["azul"], linewidth=1.6,
        )
        eixo_bits.set_title(
            "1. Sequência digital de entrada\n"
            "Cada degrau representa um bit transmitido: nível baixo = 0, nível alto = 1.",
            loc="left", fontsize=10, color=self.CORES["marinho"],
        )
        eixo_bits.set_xlabel("Tempo da sequência (s)")
        eixo_bits.set_ylabel("Valor do bit")
        eixo_bits.set_ylim(-0.2, 1.2)
        eixo_bits.set_yticks([0, 1], labels=["0", "1"])
        eixo_bits.set_xlim(0, quantidade_simbolos / rs)
        eixo_bits.grid(True, alpha=0.35, color=self.CORES["grade"])

        portadora = np.exp(
            1j * 2 * np.pi * configuracao.frequencia_portadora * tempo
        )
        sinal = np.real(resultado.envelope[:quantidade_amostras] * portadora)
        recebido = np.real(
            self.simulador.adicionar_ruido(resultado.envelope[:quantidade_amostras], k)
            * portadora
        )
        eixo_sinal.plot(
            tempo, sinal, label="Sinal modulado ideal (sem ruído)",
            color=self.CORES["azul"], linewidth=1.4,
        )
        eixo_sinal.plot(
            tempo, recebido,
            label=f"Sinal recebido + AWGN (Eb/N₀ = {configuracao.eb_n0_db:g} dB)",
            color=self.CORES["ambar"], alpha=0.75, linewidth=0.9,
        )
        eixo_sinal.set_title(
            f"2. Forma de onda passabanda — {resultado.esquema.nome}\n"
            "Sinal real = envelope complexo × portadora; a curva laranja inclui ruído gaussiano.",
            loc="left", fontsize=10, color=self.CORES["marinho"],
        )
        eixo_sinal.set_xlabel("Tempo (s)")
        eixo_sinal.set_ylabel("Amplitude instantânea")
        eixo_sinal.set_xlim(0, quantidade_simbolos / rs)
        eixo_sinal.legend(loc="upper right", fontsize=8, frameon=True)
        eixo_sinal.grid(True, alpha=0.35, color=self.CORES["grade"])
        self.canvases["onda"].draw_idle()

    def _desenhar_constelacao(self, resultado: Resultado) -> None:
        eixo = self.eixos["constelacao"][0]
        eixo.clear()
        pontos_distintos = np.unique(
            np.round(resultado.simbolos.real, 10)
            + 1j * np.round(resultado.simbolos.imag, 10)
        )
        eixo.scatter(
            resultado.simbolos.real,
            resultado.simbolos.imag,
            s=34,
            alpha=0.55,
            color=self.CORES["azul"],
            label="Símbolos da sequência (repetições sobrepostas)",
        )
        eixo.scatter(
            pontos_distintos.real,
            pontos_distintos.imag,
            s=100,
            facecolors="none",
            edgecolors=self.CORES["ambar"],
            linewidths=1.6,
            label=f"Estados distintos observados ({len(pontos_distintos)})",
        )
        eixo.axhline(0, color="gray", linewidth=0.7)
        eixo.axvline(0, color="gray", linewidth=0.7)
        eixo.set_title(
            f"Constelação I/Q — {resultado.esquema.nome}\n"
            "Cada coordenada representa um símbolo: horizontal = I (em fase), "
            "vertical = Q (quadratura).",
            loc="left", fontsize=10, color=self.CORES["marinho"],
        )
        eixo.set_xlabel("Componente em fase, I (amplitude normalizada)")
        eixo.set_ylabel("Componente em quadratura, Q (amplitude normalizada)")
        eixo.set_aspect("equal", adjustable="datalim")
        eixo.grid(True, alpha=0.35, color=self.CORES["grade"])
        eixo.legend(loc="best", fontsize=8, frameon=True)
        if resultado.esquema.nome == "BFSK":
            observacao = (
                "BFSK: I/Q mostra a representação em duas funções ortogonais "
                "(tons), não uma constelação PSK."
            )
        elif resultado.esquema.tipo == "diferencial":
            observacao = (
                "Modulação diferencial: a informação está na mudança entre "
                "símbolos consecutivos; os estados dependem do histórico."
            )
        else:
            observacao = (
                "Os círculos contornados destacam os estados distintos presentes "
                "na sequência aleatória."
            )
        eixo.text(
            0.02, 0.02, observacao, transform=eixo.transAxes,
            fontsize=8, color=self.CORES["texto"],
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        )
        self.canvases["constelacao"].draw_idle()

    def _desenhar_dashboard(self) -> None:
        eixo_rs, eixo_eficiencia = self.eixos["dashboard"]
        eixo_rs.clear()
        eixo_eficiencia.clear()
        dados = [
            (
                esquema,
                self.simulador.metricas(
                    esquema, self.simulador.gerar_resultado(esquema)
                ),
            )
            for esquema in ESQUEMAS
        ]
        nomes = [esquema.nome for esquema, _ in dados]
        x = np.arange(len(nomes))
        barras_rs = eixo_rs.bar(
            x, [item["Rs"] for _, item in dados], color=self.CORES["azul"]
        )
        eixo_rs.bar_label(barras_rs, fmt="%.0f", padding=2, fontsize=7)
        eixo_rs.set_ylabel("Símbolos por segundo (baud)")
        eixo_rs.set_title(
            "1. Taxa de símbolos — Rs = Rb/k\n"
            "Para o mesmo Rb, mais bits por símbolo reduzem a quantidade de símbolos por segundo.",
            loc="left", fontsize=10, color=self.CORES["marinho"],
        )
        eixo_rs.set_xticks(x, nomes, rotation=25, ha="right", fontsize=8)
        eixo_rs.grid(axis="y", alpha=0.35, color=self.CORES["grade"])
        barras_eficiencia = eixo_eficiencia.bar(
            x, [item["eficiencia"] for _, item in dados], color=self.CORES["verde"]
        )
        eixo_eficiencia.bar_label(
            barras_eficiencia, fmt="%.2f", padding=2, fontsize=7
        )
        eixo_eficiencia.set_ylabel("Eficiência estimada (bit/s/Hz)")
        eixo_eficiencia.set_title(
            "2. Eficiência espectral estimada — η ≈ k/(1+α)\n"
            "Estimativa de quantos bits cabem por segundo e por hertz; "
            "valores maiores indicam uso mais eficiente da banda.",
            loc="left", fontsize=10, color=self.CORES["marinho"],
        )
        eixo_eficiencia.set_xticks(x, nomes, rotation=25, ha="right", fontsize=8)
        eixo_eficiencia.grid(axis="y", alpha=0.35, color=self.CORES["grade"])
        eixo_eficiencia.set_xlabel("Técnica de modulação")
        self.canvases["dashboard"].draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador de modulacoes digitais com GUI e modo de graficos."
    )
    parser.add_argument("--bits", type=int, default=1200, help="Quantidade de bits.")
    parser.add_argument("--rb", type=float, default=1000.0, help="Taxa de bits em bit/s.")
    parser.add_argument("--fc", type=float, default=5000.0, help="Portadora em Hz.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Roll-off entre 0 e 1.")
    parser.add_argument("--ebn0", type=float, default=10.0, help="Eb/N0 em dB.")
    parser.add_argument("--seed", type=int, default=42, help="Semente pseudoaleatoria.")
    parser.add_argument(
        "--scheme",
        choices=[esquema.nome for esquema in ESQUEMAS],
        default="QPSK",
        help="Tecnica inicialmente selecionada.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Abre os graficos Matplotlib fora da interface Tkinter.",
    )
    args = parser.parse_args()
    configuracao = Configuracao(
        quantidade_bits=args.bits,
        taxa_bits=args.rb,
        frequencia_portadora=args.fc,
        roll_off=args.alpha,
        eb_n0_db=args.ebn0,
        semente=args.seed,
    )
    if args.cli:
        SimuladorModulacao(configuracao).executar(args.scheme)
        return
    raiz = tk.Tk()
    JanelaSimulador(raiz, configuracao, args.scheme)
    raiz.mainloop()


if __name__ == "__main__":
    main()
