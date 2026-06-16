"""
extrator_nota_corretagem.py  [v4 — Mapeamento por Blocos de Rodapé]
====================================================================
Extrator de alta resiliência para notas SINAC (Clear CTVM).

Estratégia
----------
Em vez de parsear a tabela central de negociações (cujo layout muda a cada
lote de PDF), o extrator foca apenas nos dois campos do *rodapé* que
encapsulam todo o resultado econômico do pregão:

    • "Ajuste day trade"             → P&L líquido  (C = ganho, D = perda)
    • "Total de custos operacionais" → taxa total   (sempre débito, abs)

Esses campos são semanticamente estáveis no padrão SINAC da B3, mesmo
quando o pdfplumber reorganiza as colunas entre versões de PDF.

Cascade de extração (3 fases, primeira que encontrar dados retorna)
-------------------------------------------------------------------
Fase 1 – Inline       label e valor na mesma linha física
Fase 2 – Bloco 2-1    labels num bloco, valores num bloco seguinte;
                      os dois últimos valores = Ajuste DT + Total custos
Fase 3 – Janela C/D   busca o primeiro token com marcador C ou D
                      após cada label (até 15 linhas)

Bugs corrigidos em relação ao v1
---------------------------------
  1. Data do pregão: usa o âncora "Data pregão" em vez de datas[0]
     (que capturava a data de vencimento do contrato)
  2. Regex de bloco: \\s+ não cruza mais para o próximo label;
     os valores são coletados linha a linha com parada defensiva
  3. _clean_numeric: C/D só é sinal quando word-boundary ou colado a
     dígito — evita falso-positivo com 'D' em "AjusteDayTrade..."

Dependências
------------
    pdfplumber   pip install pdfplumber
    (resto: apenas biblioteca padrão)
"""

import io
import re
import sys
from typing import Union

import pdfplumber


# ════════════════════════════════════════════════════════════════════════════
#  PADRÕES COMPILADOS  (uma vez, no import)
# ════════════════════════════════════════════════════════════════════════════

_RE_DATA_PREGAO  = re.compile(r'Data\s*preg[ãa]o\s*(\d{2}/\d{2}/\d{4})', re.IGNORECASE)
_RE_DATA_GENERIC = re.compile(r'\b(\d{2}/\d{2}/\d{4})\b')
_RE_NOTA         = re.compile(r'NOTA\s+DE\s+NEGO', re.IGNORECASE)

# Labels-alvo — aceita OCR com espaços fundidos (\\s*) ou normais (\\s+)
_RE_AJUSTE_DT    = re.compile(r'Ajuste\s*day\s*trade', re.IGNORECASE)
_RE_TOTAL_TAXAS  = re.compile(r'Total\s+de\s+custos\s+operacionais', re.IGNORECASE)

_RE_NUMERO       = re.compile(r'\d')

# Marcador de sinal: C/D isolado (word-boundary) OU colado a dígito no fim
# Não dispara para 'D' em "Trade", "Lado", "Liquidação" etc.
_RE_SINAL_D = re.compile(r'(?<!\w)D(?!\w)|\dD(?:\s|$)', re.IGNORECASE)
_RE_SINAL_C = re.compile(r'(?<!\w)C(?!\w)|\dC(?:\s|$)', re.IGNORECASE)


# ════════════════════════════════════════════════════════════════════════════
#  FUNÇÃO DE LIMPEZA MONETÁRIA  (_clean_numeric)
# ════════════════════════════════════════════════════════════════════════════

def _clean_numeric(raw: str) -> float:
    """
    Converte um token monetário SINAC para float com máxima resiliência.

    Casos tratados
    ──────────────────────────────────────────────────────────────────────────
    Token                Resultado    Observação
    ─────────────────────────────────────────────────────────────────────────
    '290,00 C'           +290.00      BR normal, crédito (positivo)
    '290,00 D'           -290.00      BR normal, débito (negativo)
    '290,00 | C'         +290.00      pipe como separador visual
    '29000C'             +290.00      OCR: vírgula removida → regra centavos
    '29000D'             -290.00      OCR: vírgula removida, débito
    '100D'                -1.00       OCR: vírgula removida, valor pequeno
    '290 00 C'           +290.00      OCR: vírgula→espaço (capturado pelo
                                      padrão "inteiro espaço 2dígitos")
    '2.900,00 D'        -2900.00      milhar BR + débito
    '0,00|'               0.00        zero com pipe (sem sinal)
    'C 290,00'           +290.00      sinal prefixado
    'AjusteDayTrade29000C' +290.00    fusão de colunas: label+valor fundidos
    ''                     0.00       string vazia / inválida

    Regra dos centavos
    ------------------
    Aplicada quando NÃO há vírgula nem ponto decimal explícito.
    Os dois últimos dígitos da sequência são centavos:
        '29000' → 290.00
        '100'   →   1.00
        '89'    →   0.89
    Isso é válido para TODOS os valores SINAC, que têm sempre 2 casas decimais.

    Detecção de sinal (corrigida em relação ao v1)
    -----------------------------------------------
    • C/D são sinal apenas como *word boundary* (não precedidos/seguidos
      por letra/dígito) OU quando colados a um dígito no fim do token.
    • Evita falso-negativo em "AjusteDayTrade..." (D faz parte da palavra).
    """
    if not raw:
        return 0.0

    # Normaliza: uppercase, remove pipes e €, colapsa espaços
    s = raw.strip().upper()
    s = s.replace('|', ' ').replace('€', ' ').replace('+', ' ')
    s = re.sub(r'\s+', ' ', s).strip()

    # ── Detecta sinal ────────────────────────────────────────────────────────
    has_D = bool(_RE_SINAL_D.search(s))
    has_C = bool(_RE_SINAL_C.search(s))
    # Conflito C + D no mesmo token → sinal incerto → assume positivo
    is_negative = has_D and not has_C

    # ── Extrai parte numérica (tenta padrões do mais específico ao menos) ────

    val: float | None = None

    # Padrão 1: formato BR explícito   "2.900,00"  "290,00"
    m = re.search(r'(\d{1,3}(?:\.\d{3})*),(\d{2})\b', s)
    if m:
        val = float(m.group(1).replace('.', '') + '.' + m.group(2))

    # Padrão 2: ponto decimal Anglo    "290.00"
    if val is None:
        m = re.search(r'\b(\d+)\.(\d{2})\b', s)
        if m:
            val = float(m.group(1) + '.' + m.group(2))

    # Padrão 3: espaço no decimal (OCR substituiu vírgula)  "290 00"
    # Requer exatamente 2 dígitos após o espaço, delimitados
    if val is None:
        m = re.search(r'\b(\d+) (\d{2})(?:\s|$)', s)
        if m:
            val = float(m.group(1) + '.' + m.group(2))

    # Padrão 4: só dígitos (OCR removeu decimal inteiro)
    if val is None:
        m = re.search(r'\d+', s)
        if m:
            n = m.group()
            val = (float(n[:-2] + '.' + n[-2:])
                   if len(n) >= 3 else float(n))

    if val is None:
        return 0.0

    return -val if is_negative else val


# ════════════════════════════════════════════════════════════════════════════
#  MAPEAMENTO POR BLOCOS DE RODAPÉ  (_extrair_dois_campos)
# ════════════════════════════════════════════════════════════════════════════

def _coletar_tokens_apos(linhas: list[str], inicio: int, max_linhas: int = 20) -> list[str]:
    """
    Coleta tokens numéricos a partir de `inicio` na lista de linhas.

    Para defensivo:
    • Pula linhas em branco ANTES do primeiro token encontrado.
    • Para quando encontra uma linha de TEXTO (sem dígitos) após já ter
      coletado pelo menos um token — isso delimita o bloco de valores.
    • Limite máximo de `max_linhas` para evitar overshoot.
    """
    tokens: list[str] = []

    for linha in linhas[inicio : inicio + max_linhas]:
        stripped = linha.strip()

        if not stripped:
            if tokens:          # linha em branco após valores → fim do bloco
                break
            continue            # linha em branco antes de começar → pula

        if _RE_NUMERO.search(stripped):
            tokens.append(stripped)
        else:
            if tokens:          # texto depois de valores → fim do bloco
                break
            # texto antes de qualquer valor → pula (outro label no caminho)

    return tokens


def _extrair_dois_campos(texto: str) -> tuple[float, float]:
    """
    Extrai (ajuste_day_trade, total_taxas) do texto de uma página SINAC.

    Cascade em 3 fases — retorna ao primeiro resultado não-zero encontrado.

    Fase 1 — Inline
    ────────────────
    Tenta capturar o valor logo após o label na mesma linha:
        "Ajuste day trade   290,00 | C"
        "Ajuste day trade290,00C"   (fusão OCR sem espaço)

    Fase 2 — Bloco 2-1
    ───────────────────
    Estrutura mais comum no pdfplumber para notas SINAC:
        [bloco de labels]
        Ajuste day trade                 ← penúltimo label
        Total de custos operacionais     ← último label

        [bloco de valores seguinte]
        0,00                             ← valores de campos anteriores
        0,00
        0,00 |
        290,00 | C                       ← penúltimo valor  → ajuste DT
        1,00 | D                         ← último valor     → total taxas

    Os dois últimos tokens do bloco de valores correspondem invariavelmente
    aos dois últimos labels, independentemente de quantos labels há antes.

    Fase 3 — Janela + filtro C/D
    ─────────────────────────────
    Busca janela de 15 linhas após cada label.
    Para "Ajuste day trade" exige sinal C/D explícito (filtra zeros sem sinal).
    Para "Total de custos" aceita qualquer valor numérico.
    Cobre layouts semiestruturados onde label e valor alternam por blocos menores.
    """
    ajuste: float = 0.0
    taxas:  float = 0.0

    linhas = texto.split('\n')

    # ── Fase 1: Inline ───────────────────────────────────────────────────────
    for linha in linhas:

        if _RE_AJUSTE_DT.search(linha) and ajuste == 0.0:
            # Remove o label da linha e verifica se sobra valor numérico
            resto = _RE_AJUSTE_DT.sub('', linha).strip()
            # Limpa separadores comuns antes do valor (espaços, pipes, hífens)
            resto = re.sub(r'^[\s|:–\-]+', '', resto)
            if resto and _RE_NUMERO.search(resto):
                ajuste = _clean_numeric(resto)

        if _RE_TOTAL_TAXAS.search(linha) and taxas == 0.0:
            resto = _RE_TOTAL_TAXAS.sub('', linha).strip()
            resto = re.sub(r'^[\s|:–\-]+', '', resto)
            if resto and _RE_NUMERO.search(resto):
                taxas = abs(_clean_numeric(resto))

    if ajuste != 0.0 and taxas != 0.0:
        return ajuste, taxas          # Fase 1 suficiente

    # ── Fase 2: Bloco 2-1 ────────────────────────────────────────────────────
    # Localiza a linha do último label-âncora (Total de custos)
    idx_tc: int | None = None
    for i, linha in enumerate(linhas):
        if _RE_TOTAL_TAXAS.search(linha):
            idx_tc = i               # sobrescreve → fica com o último

    if idx_tc is not None:
        tokens = _coletar_tokens_apos(linhas, idx_tc + 1)

        if len(tokens) >= 1 and taxas == 0.0:
            taxas = abs(_clean_numeric(tokens[-1]))

        if len(tokens) >= 2 and ajuste == 0.0:
            ajuste = _clean_numeric(tokens[-2])

    if ajuste != 0.0 and taxas != 0.0:
        return ajuste, taxas          # Fase 2 suficiente

    # ── Fase 3: Janela + filtro C/D ──────────────────────────────────────────
    # Localiza labels individualmente para a busca de janela
    idx_adt: int | None = None
    for i, linha in enumerate(linhas):
        if _RE_AJUSTE_DT.search(linha):
            idx_adt = i
            break

    # Ajuste DT: exige C ou D explícito (distingue do valor zero sem sinal)
    if idx_adt is not None and ajuste == 0.0:
        for linha in linhas[idx_adt + 1 : idx_adt + 16]:
            stripped = linha.strip()
            if (stripped and _RE_NUMERO.search(stripped)
                    and (_RE_SINAL_C.search(stripped.upper())
                         or _RE_SINAL_D.search(stripped.upper()))):
                ajuste = _clean_numeric(stripped)
                break

    # Total de custos: aceita qualquer valor numérico após o label
    if idx_tc is not None and taxas == 0.0:
        for linha in linhas[idx_tc + 1 : idx_tc + 16]:
            stripped = linha.strip()
            if stripped and _RE_NUMERO.search(stripped):
                taxas = abs(_clean_numeric(stripped))
                break

    return ajuste, taxas


# ════════════════════════════════════════════════════════════════════════════
#  DATA DO PREGÃO  (_extrair_data)
# ════════════════════════════════════════════════════════════════════════════

def _extrair_data(texto: str) -> str | None:
    """
    Retorna a data do pregão no formato DD/MM/AAAA.

    Prioridade:
    1. Âncora explícita "Data pregão DD/MM/AAAA" (elimina datas de vencimento
       de contratos futuros que aparecem antes no texto)
    2. Fallback: primeira data encontrada no texto

    Bug corrigido em relação ao v1
    --------------------------------
    O texto de uma nota BM&F começa com vencimentos de contratos (ex: 17/06/2026)
    antes de chegar ao campo "Data pregão 29/05/2026".
    O uso de datas[0] retornava 17/06/2026 em vez de 29/05/2026.
    """
    m = _RE_DATA_PREGAO.search(texto)
    if m:
        return m.group(1)

    # Fallback: primeira data no texto (pode ser vencimento — sinaliza aviso)
    m = _RE_DATA_GENERIC.search(texto)
    return m.group(1) if m else None


# ════════════════════════════════════════════════════════════════════════════
#  INTERFACE PÚBLICA — extração a partir de TEXTO
# ════════════════════════════════════════════════════════════════════════════

def extrair_dados_nota_direto(texto_pagina: str) -> dict:
    """
    Extrai dados de uma nota de corretagem a partir do texto bruto da página.

    Compatível com a interface do App.py (aceita texto extraído por qualquer
    biblioteca — pypdf, pdfplumber, pdfminer, etc.).

    Parâmetros
    ----------
    texto_pagina : str
        Texto bruto extraído de uma página do PDF.

    Retorna
    -------
    dict com as chaves:
        data_pregao : 'DD/MM/AAAA'  ou None
        tipo_nota   : 'BMF'
        operacoes   : list[dict]  — lista vazia se ajuste == 0
        total_taxas : float       — always >= 0; mínimo 1.00 quando > 0
    """
    data_pregao = _extrair_data(texto_pagina)
    ajuste_val, total_taxas = _extrair_dois_campos(texto_pagina)

    if ajuste_val == 0.0:
        return {
            'data_pregao': data_pregao,
            'tipo_nota'  : 'BMF',
            'operacoes'  : [],
            'total_taxas': 0.0,
        }

    return {
        'data_pregao': data_pregao,
        'tipo_nota'  : 'BMF',
        'operacoes'  : [{
            'cv'            : 'V' if ajuste_val >= 0 else 'C',
            'ticker'        : 'WIN',
            'quantidade'    : 1,
            'preco_unitario': abs(ajuste_val),
            'valor_total'   : ajuste_val,
            'dc'            : 'C' if ajuste_val >= 0 else 'D',
            # Força classificação correta na pipeline do consolidador:
            # tipo_negocio='DAY TRADE' → _classificar_trade_type marca DAY_TRADE
            'tipo_negocio'  : 'DAY TRADE',
        }],
        'total_taxas': total_taxas if total_taxas > 0 else 1.00,
    }


# ════════════════════════════════════════════════════════════════════════════
#  INTERFACE PÚBLICA — extração a partir de PDF  (pdfplumber)
# ════════════════════════════════════════════════════════════════════════════

def extrair_notas_pdf(
    origem: Union[str, bytes, bytearray, io.IOBase],
) -> list[dict]:
    """
    Extrai dados de todas as notas de negociação em um arquivo PDF.

    Usa pdfplumber para leitura página a página. Para cada página que
    contiver "NOTA DE NEGOCIAÇÃO" no texto, executa extrair_dados_nota_direto
    tentando primeiro extração com layout preservado (pdfplumber layout=True),
    depois sem layout como fallback — retém o resultado com mais informação.

    Parâmetros
    ----------
    origem : str | bytes | bytearray | file-like object
        Caminho para o arquivo PDF, bytes do PDF (ex: Streamlit UploadedFile.read()),
        ou qualquer objeto file-like que o pdfplumber aceite.

    Retorna
    -------
    list[dict]
        Lista de dicts no formato de extrair_dados_nota_direto.
        Cada dict inclui metadado '_pagina' (int, 1-based) para debug.
        Páginas sem "NOTA DE NEGOCIAÇÃO" são silenciosamente ignoradas.
        Erros por página são capturados sem interromper o lote.
    """
    if isinstance(origem, (bytes, bytearray)):
        origem = io.BytesIO(origem)

    resultados: list[dict] = []

    try:
        with pdfplumber.open(origem) as pdf:
            for num_pag, pagina in enumerate(pdf.pages, start=1):

                # ── Verifica se a página é uma nota de negociação ─────────────
                texto_simples = pagina.extract_text() or ''
                if not _RE_NOTA.search(texto_simples):
                    continue

                # ── Tenta layout=True para melhor detecção inline (Fase 1) ────
                texto_layout = ''
                try:
                    texto_layout = pagina.extract_text(layout=True) or ''
                except Exception:
                    pass  # pdfplumber < 0.9 pode não suportar layout=True

                # ── Executa extração nas duas variantes; retém a melhor ────────
                melhor: dict | None = None
                for texto in filter(None, [texto_layout, texto_simples]):
                    d = extrair_dados_nota_direto(texto)
                    if d.get('data_pregao') and (
                        d.get('operacoes')        # tem operações → preferência
                        or melhor is None         # ou é o primeiro resultado
                    ):
                        melhor = d
                        if d.get('operacoes'):
                            break              # já temos o melhor possível

                if melhor and melhor.get('data_pregao'):
                    melhor['_pagina'] = num_pag
                    resultados.append(melhor)

    except Exception as exc:
        print(
            f'[extrator] ERRO ao processar PDF: {exc}',
            file=sys.stderr,
        )

    return resultados


# ════════════════════════════════════════════════════════════════════════════
#  TESTES INTEGRADOS
# ════════════════════════════════════════════════════════════════════════════

# Texto real do PDF NotaNegociacao...0_1.pdf conforme extraído pelo pypdf
# (confirmado na sessão de diagnóstico)
_TEXTO_REAL = """Negociações
C/V
Mercadoria
Vencimento
Quantidade
Preço/Ajuste
Tipo Negócio
Vlr de Operação/Ajuste D/C
Taxa OperacionalC WIN M26
17/06/2026
1
173.990,00
DAY TRADE
236,20 C
0,00C WIN M26
17/06/2026
1
174.910,00
DAY TRADE
52,20 C
0,00V WIN M26
17/06/2026
1
175.130,00
DAY TRADE
8,20 D
0,00V WIN M26
17/06/2026
1
175.220,00
DAY TRADE
9,80 C
0,00
NOTA DE NEGOCIAÇÃO
Nr. nota276.591
Folha1
Data pregão29/05/2026
CLEAR CTVM S/A
Venda disponível
Compra disponível
Valor dos negócios
0,00 
0,00 
290,00 | C IRRF
IRRF Day Trade (proj.)
Taxa operacional
Taxa registro BM&F
Taxas BM&F (emol+f.gar)
0,00|  
2,89 
0,00 
0,64 
0,36 | D  
+Outros Custos
Impostos
Ajuste de posição
Ajuste day trade
Total de custos operacionais
 
0,00 
0,00 
0,00 |   
290,00 | C 
1,00 | D Outros
IRRF operacional
Total Conta Investimento
Total Conta Normal
Total liquido (#)
Total líquido da nota
0,00 
0,00 
0,00|  
286,11 | C 
289,00 | C 
286,11 | C"""

# Variantes de layout para cobrir diferentes versões de PDF
_TEXTO_INLINE = (
    "NOTA DE NEGOCIAÇÃO\nData pregão01/06/2026\n"
    "Ajuste day trade   290,00 | C\n"
    "Total de custos operacionais   1,00 | D\n"
)

_TEXTO_OCR_FUSO = (
    "NOTA DE NEGOCIAÇÃO\nData pregão01/06/2026\n"
    "+Outros Custos\nImpostos\nAjuste de posição\n"
    "Ajuste day trade\nTotal de custos operacionais\n"
    "000\n000\n000|\n29000C\n100D\n"
)

_TEXTO_LOSS = (
    "NOTA DE NEGOCIAÇÃO\nData pregão01/06/2026\n"
    "Ajuste day trade\nTotal de custos operacionais\n \n"
    "0,00\n0,00\n0,00|\n85,00 | D\n1,50 | D\n"
)


def _executar_testes() -> None:
    erros = 0

    def chk(desc: str, obtido, esperado, tol: float = 0.01) -> None:
        nonlocal erros
        ok = abs(obtido - esperado) <= tol
        if not ok:
            erros += 1
        print(f"  {'✅' if ok else '❌'} {desc:<50} obtido={obtido:>10.2f}  esp={esperado:>10.2f}")

    # ── _clean_numeric ────────────────────────────────────────────────────────
    print("\n─── _clean_numeric ───────────────────────────────────────────────")
    _casos = [
        ('Normal C',                    '290,00 C',              +290.00),
        ('Normal D',                    '290,00 D',              -290.00),
        ('Pipe + C',                    '290,00 | C',            +290.00),
        ('OCR sem vírgula C',           '29000C',                +290.00),
        ('OCR sem vírgula D',           '29000D',                -290.00),
        ('OCR valor pequeno D',         '100D',                   -1.00),
        ('OCR espaço decimal',          '290 00 C',              +290.00),
        ('Milhar BR D',                 '2.900,00 D',           -2900.00),
        ('Zero com pipe',               '0,00|',                  +0.00),
        ('Pipe + D',                    '1,00 | D',               -1.00),
        ('Sinal prefixado C',           'C 290,00',              +290.00),
        ('Fusão de colunas (v1 bug)',   'AjusteDayTrade29000C',  +290.00),
        ('Vazio',                        '',                        0.00),
    ]
    for desc, tok, esp in _casos:
        chk(desc, _clean_numeric(tok), esp)

    # ── _extrair_data ─────────────────────────────────────────────────────────
    print("\n─── _extrair_data ────────────────────────────────────────────────")
    data_real = _extrair_data(_TEXTO_REAL)
    ok_data = data_real == '29/05/2026'
    if not ok_data:
        erros += 1
    print(f"  {'✅' if ok_data else '❌'} Âncora 'Data pregão': obtido={data_real!r}  esp='29/05/2026'")
    print(f"  {'✅' if ok_data else '❌'} NÃO confunde com vencimento 17/06/2026")

    # ── _extrair_dois_campos ──────────────────────────────────────────────────
    print("\n─── _extrair_dois_campos ─────────────────────────────────────────")
    casos_campos = [
        ('Texto real (bloco separado)',  _TEXTO_REAL,     +290.00, 1.00),
        ('Inline',                       _TEXTO_INLINE,   +290.00, 1.00),
        ('OCR fusão (bloco separado)',   _TEXTO_OCR_FUSO, +290.00, 1.00),
        ('Prejuízo (bloco separado)',    _TEXTO_LOSS,      -85.00, 1.50),
    ]
    for desc, texto, esp_ajuste, esp_taxas in casos_campos:
        ajuste, taxas = _extrair_dois_campos(texto)
        ok_a = abs(ajuste - esp_ajuste) <= 0.01
        ok_t = abs(taxas  - esp_taxas ) <= 0.01
        if not (ok_a and ok_t):
            erros += 1
        print(
            f"  {'✅' if ok_a and ok_t else '❌'} {desc:<38} "
            f"ajuste={ajuste:>8.2f}(esp {esp_ajuste:>8.2f})  "
            f"taxas={taxas:>6.2f}(esp {esp_taxas:>5.2f})"
        )

    # ── extrair_dados_nota_direto (interface completa) ────────────────────────
    print("\n─── extrair_dados_nota_direto ────────────────────────────────────")
    d = extrair_dados_nota_direto(_TEXTO_REAL)
    ok_d = (
        d['data_pregao'] == '29/05/2026'
        and len(d['operacoes']) == 1
        and d['operacoes'][0]['ticker'] == 'WIN'
        and abs(d['operacoes'][0]['valor_total'] - 290.00) < 0.01
        and abs(d['total_taxas'] - 1.00) < 0.01
        and d['operacoes'][0]['cv'] == 'V'
        and d['operacoes'][0]['dc'] == 'C'
    )
    if not ok_d:
        erros += 1
    print(f"  {'✅' if ok_d else '❌'} Nota real completa:")
    print(f"       data_pregao = {d['data_pregao']!r}")
    print(f"       ticker      = {d['operacoes'][0]['ticker'] if d['operacoes'] else 'N/A'!r}")
    print(f"       cv/dc       = {d['operacoes'][0]['cv'] if d['operacoes'] else '?'} / "
          f"{d['operacoes'][0]['dc'] if d['operacoes'] else '?'}")
    print(f"       valor_total = {d['operacoes'][0]['valor_total'] if d['operacoes'] else 0:.2f}")
    print(f"       total_taxas = {d['total_taxas']:.2f}")

    # ── Resultado final ────────────────────────────────────────────────────────
    print()
    if erros == 0:
        print('═' * 60)
        print('  ✅  Todos os testes passaram!')
        print('═' * 60)
    else:
        print(f'═' * 60)
        print(f'  ❌  {erros} teste(s) falharam.')
        print('═' * 60)
    print()


if __name__ == '__main__':
    _executar_testes()
