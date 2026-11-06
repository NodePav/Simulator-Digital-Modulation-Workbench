# Simulador e visualizador de modulacoes digitais

Aplicativo didatico em Python com interface Tkinter para ASK/OOK, BFSK, M-PSK
(8-PSK), BPSK, QPSK, DPSK, DQPSK, OQPSK, pi/4-DQPSK, MSK, 16-QAM e 16-QAM
diferencial. A janela tem uma lista para escolher a tecnica, campos para editar
bits e parametros, um memorial de calculo e abas com forma de onda, constelacao
I/Q e comparacao das 12 tecnicas.

## Executar no terminal do VS Code (Windows)

**Execução mais simples:** dê dois cliques em `iniciar_simulador.bat`. O
arquivo cria o ambiente virtual `.venv` se necessário, instala/atualiza as
dependências e abre a interface.

Para executar pelo terminal do VS Code:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python simulador.py
```

## Usar e publicar a versão web

A versão web está no arquivo único [index.html](./index.html). Ela roda
diretamente no navegador (celular ou computador), sem Python, instalação,
servidor próprio ou bibliotecas externas. Para testar, abra `index.html` no
navegador.

Para criar um link público de acesso, publique esse arquivo em uma hospedagem
estática. Por exemplo:

1. Abra uma conta no Netlify e entre em **Netlify Drop** (`https://app.netlify.com/drop`).
2. Arraste o arquivo `index.html` para a área de publicação.
3. Aguarde o deploy e copie o endereço HTTPS gerado. Esse será o link para
   compartilhar.

Alternativamente, envie `index.html` para um repositório GitHub e ative
**Settings → Pages** para publicar pelo GitHub Pages. O link público só é
criado pela plataforma após a publicação; este projeto não tem acesso às suas
contas de hospedagem.

O simulador web faz os cálculos localmente no navegador e não envia os bits
gerados para um servidor. O endereço público depende das condições de
disponibilidade e política da hospedagem escolhida.

Na janela, basta alterar a quantidade de bits, Rb, fc, roll-off ou Eb/N0:
gráficos e métricas são recalculados automaticamente após uma breve pausa.
Também é possível trocar a técnica pela lista lateral. **Atualizar agora**
executa o recálculo imediatamente. A quantidade de bits deve ser um inteiro de
pelo menos 8.

A interface tem identidade visual MODLAB e uma aba **Ajuda** com instruções
passo a passo, explicação das métricas, leitura dos gráficos e dicas para
resolver entradas inválidas ou limitações de visualização.

Na barra **Configuração**, o quadro **Sobre a técnica** acompanha a seleção e
resume o significado da modulação escolhida e exemplos de aplicações comuns.
As aplicações são referências introdutórias, não uma lista exclusiva de usos.

## Como interpretar os gráficos

- **Forma de onda:** o primeiro painel mostra os bits (0/1) ao longo do tempo.
  No segundo, a linha azul é o sinal modulado ideal e a linha laranja é a
  realização com ruído AWGN; a legenda informa o Eb/N0 aplicado. São exibidos
  até 12 períodos de símbolo, com eixos de tempo (s) e amplitude instantânea.
- **Constelação I/Q:** o eixo horizontal é I (componente em fase) e o vertical
  é Q (quadratura), em amplitudes normalizadas. Os pontos azuis são os símbolos
  gerados; círculos contornados indicam estados distintos observados. Símbolos
  repetidos ficam sobrepostos. Para BFSK, I/Q representa duas funções
  ortogonais (tons); nas técnicas diferenciais, a informação está na mudança
  entre símbolos.
- **Dashboard — taxa de símbolos:** cada barra mostra Rs = Rb/k em baud; o valor
  numérico aparece sobre a barra. Com Rb fixo, mais bits por símbolo resultam
  em menos símbolos transmitidos por segundo.
- **Dashboard — eficiência espectral:** apresenta a estimativa
  η ≈ k/(1+α), em bit/s/Hz. É uma comparação didática, não uma medida real de
  desempenho nem uma garantia de largura de banda ocupada.

Os valores tambem podem ser definidos ao iniciar:

```powershell
python simulador.py --scheme "16-QAM" --bits 2000 --rb 2000 --fc 10000 --alpha 0.25 --ebn0 12
```

Para abrir a versao anterior com janelas Matplotlib independentes, use
`python simulador.py --cli`. `python simulador.py --help` lista todas as opcoes.
O grafico temporal mostra os bits, sinal modulado e uma realizacao com ruido
AWGN para o Eb/N0 informado.

## Modelo e convencoes

- `Rb` e a taxa de bits; `k = log2(M)`; `Rs = Rb/k`; `Tb = 1/Rb` e
  `Ts = 1/Rs`.
- A banda do painel usa a aproximacao didatica `B ≈ (1 + alpha) Rs`. Para FSK
  e MSK, a largura de banda real depende tambem do desvio entre tons e do
  criterio de ocupacao espectral; portanto, a aproximacao nao substitui esse
  calculo.
- Os pontos sao normalizados para energia media por simbolo `Es = 1`; assim,
  `Eb = Es/k`. O painel mostra esses valores para a sequencia gerada.
- O mapeamento de 16-QAM usa niveis I/Q Gray normalizados por `sqrt(10)`.
  A QAM diferencial implementada e uma variante didatica: amplitude instantanea
  de 16-QAM com fase diferencial acumulada; nao representa um unico padrao
  universal de DQAM.
- MSK e gerada como CPFSK com indice `h=0.5` e fase continua. O atraso do ramo
  Q em OQPSK e de meio periodo de simbolo.
- `alpha` e usado na estimativa da banda. As formas de onda usam pulsos
  retangulares para manter o exemplo autocontido; nao aplicam filtro raised
  cosine.

## Dependencias

- Python 3.10 ou superior
- NumPy
- Matplotlib
- Tkinter (incluido nas instalacoes oficiais do Python para Windows)
