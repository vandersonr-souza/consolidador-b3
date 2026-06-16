"""
consolidador_notas.py  [v3 — IN RFB nº 1585/2015]
==================================================
Lógica de negócios sobre notas de corretagem SINAC.

Correções fiscais aplicadas nesta versão
-----------------------------------------
1. Day Trade Parcial — Row Splitting
   Quando uma ordem é parcialmente Day Trade (ex.: compra 100, vende 30),
   a linha original é dividida em duas linhas físicas no DataFrame:
     • DT portion  (30 unid.)  tipo_trade = 'DAY_TRADE'
     • SW portion  (70 unid.)  tipo_trade = 'SWING_TRADE'
   Elimina a heurística de 50% que gerava classificações incorretas.

2. Isolamento Fiscal do Day Trade  (IN RFB nº 1585/2015)
   O resultado do Day Trade de ações é calculado exclusivamente pelos
   preços intraday do próprio pregão:
     resultado_dt = (PM_venda_adj_dia − PM_compra_adj_dia) × qty_dt
   O preço médio histórico da custódia NÃO é usado no Day Trade e o
   Day Trade NÃO altera o preço médio carregado para meses seguintes.
   Somente operações 'SWING_TRADE' interagem com o livro de custódia.

Pipeline
--------
    notas → ratear_taxas()
          → _classificar_trade_type()    ← Row Splitting
          → _precomputar_dt_acoes()      ← pre-cálculo DT por sessão
          → _calcular_preco_medio_e_resultado()
          → _resultado_mensal()
"""

import numpy as np
import pandas as pd
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ════════════════════════════════════════════════════════════════════════════

LIMITE_ISENCAO_SWING: float = 20_000.00   # R$ mensais — isenção ações swing
ALIQUOTA_DAYTRADE:    float = 0.20        # 20% IR Day Trade
ALIQUOTA_SWING:       float = 0.15        # 15% IR Swing Trade
IRRF_DAYTRADE:        float = 0.01        # 1% IRRF retido na fonte (Day Trade)
_RATIO_FUTURO_THRESHOLD: float = 0.05     # heurística futuros vs ações


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — RATEIO DE TAXAS
# ════════════════════════════════════════════════════════════════════════════

def ratear_taxas(nota: dict) -> pd.DataFrame:
    """
    Converte uma nota em DataFrame e distribui o total de taxas
    proporcionalmente ao volume nocional de cada operação.

    Volume nocional:
        Ações   → abs(valor_total)          (preco × qty já está na nota)
        Futuros → preco_unitario × quantidade (evita distorção pelo P&L diário)

    Colunas adicionadas
    -------------------
    tipo_ativo       'ACAO' | 'FUTURO'
    volume_nocional  base de rateio (R$)
    taxa_rateada     R$ de taxa alocado a esta linha
    taxa_por_unidade taxa_rateada / quantidade
    preco_ajustado   preço efetivo de custo (compra) ou receita líquida (venda)
    """
    df = pd.DataFrame(nota['operacoes'])
    df['data_pregao']      = pd.to_datetime(nota['data_pregao'], dayfirst=True)
    df['total_taxas_nota'] = nota['total_taxas']

    nocional = df['preco_unitario'] * df['quantidade']
    ratio    = df['valor_total'].abs() / nocional.replace(0, np.nan)
    df['tipo_ativo'] = np.where(
        ratio.fillna(1.0) < _RATIO_FUTURO_THRESHOLD, 'FUTURO', 'ACAO'
    )
    # Patch fiscal: notas BM&F (futuros) têm preco_unitario = abs(ajuste_val),
    # o que faz ratio ≈ 1,0 e dispara detecção errada como ACAO.
    # O campo tipo_nota='BMF' é a fonte autoritativa — prevalece sobre a heurística.
    if nota.get('tipo_nota') == 'BMF':
        df['tipo_ativo'] = 'FUTURO' 

    df['volume_nocional'] = np.where(
        df['tipo_ativo'] == 'FUTURO', nocional, df['valor_total'].abs()
    )

    total_vol = df['volume_nocional'].sum()
    df['taxa_rateada'] = (
        nota['total_taxas'] * (df['volume_nocional'] / total_vol)
        if total_vol > 0 else 0.0
    )
    df['taxa_por_unidade'] = (
        df['taxa_rateada'] / df['quantidade'].replace(0, np.nan)
    )
    df['preco_ajustado'] = np.where(
        df['cv'] == 'C',
        df['preco_unitario'] + df['taxa_por_unidade'],
        df['preco_unitario'] - df['taxa_por_unidade'],
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — CLASSIFICAÇÃO DAY TRADE / SWING TRADE  (com Row Splitting)
# ════════════════════════════════════════════════════════════════════════════

def _criar_linha_split(
    row: pd.Series,
    nova_qty: int,
    tipo_trade: str,
) -> dict:
    """
    Cria um dict representando uma linha derivada de um split de row.

    Campos escalados proporcionalmente à nova quantidade:
        quantidade, taxa_rateada, valor_total, volume_nocional

    Campos por unidade (permanecem inalterados):
        preco_unitario, preco_ajustado, taxa_por_unidade, dc, …
    """
    ratio = nova_qty / row['quantidade']
    d = row.to_dict()
    d['quantidade']   = nova_qty
    d['tipo_trade']   = tipo_trade
    d['taxa_rateada'] = row['taxa_por_unidade'] * nova_qty
    d['valor_total']  = row['valor_total'] * ratio
    if 'volume_nocional' in d:
        d['volume_nocional'] = row['volume_nocional'] * ratio
    return d


def _classificar_trade_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca 'tipo_trade' como 'DAY_TRADE' ou 'SWING_TRADE' com Row Splitting.

    Regra B3 / IN RFB nº 1585/2015
    --------------------------------
    Se no mesmo pregão um ativo tiver compras E vendas, o menor lado
    define a quantidade Day Trade (qty_dt = min(Σ compras, Σ vendas)).

    Row Splitting
    -------------
    Quando qty_dt < lado total da ordem (Day Trade parcial), a linha
    original é REMOVIDA e substituída por duas novas linhas físicas:
        • tipo_trade = 'DAY_TRADE'   com qty = porção intraday
        • tipo_trade = 'SWING_TRADE' com qty = porção restante

    O orçamento de qty_dt é alocado independentemente para compras e
    vendas — cada lado tem seu próprio saldo (saldo_c / saldo_v).
    """
    df = df.copy().reset_index(drop=True)
    df['tipo_trade'] = 'SWING_TRADE'

    idx_remover: list[int] = []
    linhas_novas: list[dict] = []

    for (_, _ticker), grp_idx in df.groupby(
        ['data_pregao', 'ticker'], sort=False
    ).groups.items():
        grp   = df.loc[grp_idx]
        qty_c = grp.loc[grp['cv'] == 'C', 'quantidade'].sum()
        qty_v = grp.loc[grp['cv'] == 'V', 'quantidade'].sum()

        if qty_c == 0 or qty_v == 0:
            continue   # apenas um lado → tudo Swing

        qty_dt  = min(qty_c, qty_v)
        saldo_c = qty_dt   # orçamento DT para o lado compra
        saldo_v = qty_dt   # orçamento DT para o lado venda

        for lado, saldo_ref in [('C', 'saldo_c'), ('V', 'saldo_v')]:
            saldo = saldo_c if lado == 'C' else saldo_v

            for i in grp[grp['cv'] == lado].index:
                if saldo <= 0:
                    break
                q    = df.at[i, 'quantidade']
                dt_q = min(q, saldo)
                sw_q = q - dt_q
                saldo -= dt_q

                if sw_q == 0:
                    # Linha inteira é Day Trade — sem split
                    df.at[i, 'tipo_trade'] = 'DAY_TRADE'
                else:
                    # Day Trade parcial — split em duas linhas físicas
                    idx_remover.append(i)
                    linhas_novas.append(
                        _criar_linha_split(df.loc[i], dt_q, 'DAY_TRADE')
                    )
                    linhas_novas.append(
                        _criar_linha_split(df.loc[i], sw_q, 'SWING_TRADE')
                    )

                if lado == 'C':
                    saldo_c = saldo
                else:
                    saldo_v = saldo

    # Substitui as linhas originais pelas linhas divididas
    df = df.drop(index=idx_remover)
    if linhas_novas:
        df = pd.concat(
            [df, pd.DataFrame(linhas_novas)], ignore_index=True
        )

    # Segurança: campo 'tipo_negocio' da nota prevalece (usado em BM&F)
    # Split SW rows não têm tipo_negocio='DAY TRADE', então não são afetadas.
    if 'tipo_negocio' in df.columns:
        mask = df['tipo_negocio'].str.contains('DAY', na=False, case=False)
        df.loc[mask, 'tipo_trade'] = 'DAY_TRADE'

    return df.sort_values(
        ['data_pregao', 'ticker', 'cv']
    ).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — PREÇO MÉDIO E RESULTADO  (com isolamento fiscal DT)
# ════════════════════════════════════════════════════════════════════════════

def _precomputar_dt_acoes(df: pd.DataFrame) -> dict:
    """
    Pré-calcula o resultado por unidade de cada sessão Day Trade de ações.

    Retorna
    -------
    dict  (data_pregao, ticker) → {
        'resultado_por_unidade': float,   # pm_venda_adj − pm_compra_adj
        'pm_compra'            : float,   # média ponderada dos preços de compra DT
        'pm_venda'             : float,   # média ponderada dos preços de venda DT
    }

    Fórmula (IN RFB nº 1585/2015)
    --------------------------------
        pm_compra = Σ(preco_ajustado_c × qty_c) / Σ qty_c   (apenas linhas DT)
        pm_venda  = Σ(preco_ajustado_v × qty_v) / Σ qty_v   (apenas linhas DT)
        resultado_por_unidade = pm_venda − pm_compra
    """
    sessoes: dict = {}
    mascara = (df['tipo_trade'] == 'DAY_TRADE') & (df['tipo_ativo'] == 'ACAO')

    for (data, ticker), grp in df[mascara].groupby(
        ['data_pregao', 'ticker'], sort=False
    ):
        buys  = grp[grp['cv'] == 'C']
        sells = grp[grp['cv'] == 'V']
        if buys.empty or sells.empty:
            continue

        qty_c = buys['quantidade'].sum()
        qty_v = sells['quantidade'].sum()

        pm_c = (buys['preco_ajustado'] * buys['quantidade']).sum() / qty_c
        pm_v = (sells['preco_ajustado'] * sells['quantidade']).sum() / qty_v

        sessoes[(data, ticker)] = {
            'resultado_por_unidade': pm_v - pm_c,
            'pm_compra'            : pm_c,
            'pm_venda'             : pm_v,
        }

    return sessoes


def _calcular_preco_medio_e_resultado(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Percorre as operações em ordem cronológica aplicando as regras fiscais:

    Day Trade — ações (IN RFB nº 1585/2015)
    ----------------------------------------
    • Compra DT: resultado = 0, NÃO atualiza o preço médio histórico.
    • Venda DT : resultado = resultado_por_unidade × qty  (preços intraday).
    • O PM histórico da carteira permanece INALTERADO pelo Day Trade.

    Swing Trade — ações
    -------------------
    • Compra SW: atualiza PM ponderado da custódia.
    • Venda SW : realiza P&L = (preco_ajustado − PM) × qty;
                 PM não muda na venda.

    Futuros (ambos os tipos de trade)
    -----------------------------------
    • resultado = valor_total − taxa_rateada  (P&L de ajuste diário).
    • Posição líquida rastreada em qty (pode ser negativa = vendido).

    Retorna
    -------
    df_ops      : DataFrame enriquecido com 'preco_medio_na_op' e 'resultado_bruto'
    df_custodia : posições abertas (|qty| > 0) com preço médio e custo total
    """
    df = df.sort_values(['data_pregao', 'ticker']).copy().reset_index(drop=True)
    df['preco_medio_na_op'] = np.nan
    df['resultado_bruto']   = np.nan

    # Pré-computa sessões DT de ações
    sessoes_dt = _precomputar_dt_acoes(df)

    # Livro de posições: { ticker: { qty, avg, tipo_ativo, ultima_compra } }
    livro: dict = {}

    for i, row in df.iterrows():
        t     = row['ticker']
        tipo  = row['tipo_ativo']
        qty   = row['quantidade']
        trade = row['tipo_trade']

        if t not in livro:
            livro[t] = {
                'qty'          : 0,
                'avg'          : 0.0,
                'tipo_ativo'   : tipo,
                'ultima_compra': pd.NaT,
            }
        pos = livro[t]

        # ── FUTUROS ────────────────────────────────────────────────────────
        if tipo == 'FUTURO':
            df.at[i, 'resultado_bruto'] = row['valor_total'] - row['taxa_rateada']
            if row['cv'] == 'C':
                pos['qty'] += qty
                pos['ultima_compra'] = row['data_pregao']
            else:
                pos['qty'] -= qty
            continue

        # ── AÇÕES — DAY TRADE ───────────────────────────────────────────────
        chave_dt = (row['data_pregao'], t)

        if trade == 'DAY_TRADE':
            if row['cv'] == 'C':
                # Compra DT: sem P&L realizado, sem impacto no PM histórico
                df.at[i, 'resultado_bruto'] = 0.0

            else:
                # Venda DT: P&L calculado exclusivamente pelos preços do dia
                sess = sessoes_dt.get(chave_dt)
                if sess:
                    df.at[i, 'resultado_bruto'] = (
                        sess['resultado_por_unidade'] * qty
                    )
                    df.at[i, 'preco_medio_na_op'] = sess['pm_compra']
                else:
                    df.at[i, 'resultado_bruto'] = 0.0

        # ── AÇÕES — SWING TRADE ─────────────────────────────────────────────
        else:
            if row['cv'] == 'C':
                # Compra SW: atualiza PM ponderado
                novo_custo    = pos['qty'] * pos['avg'] + qty * row['preco_ajustado']
                pos['qty']   += qty
                pos['avg']    = novo_custo / pos['qty'] if pos['qty'] > 0 else 0.0
                pos['ultima_compra'] = row['data_pregao']
                df.at[i, 'preco_medio_na_op'] = pos['avg']
                df.at[i, 'resultado_bruto']   = 0.0

            else:
                # Venda SW: realiza P&L contra PM histórico
                pm = pos['avg']
                df.at[i, 'preco_medio_na_op'] = pm
                df.at[i, 'resultado_bruto']   = (
                    (row['preco_ajustado'] - pm) * qty
                )
                pos['qty'] = max(0, pos['qty'] - qty)
                if pos['qty'] == 0:
                    pos['avg'] = 0.0

    # ── Monta DataFrame de custódia ─────────────────────────────────────────
    registros = [
        {
            'ticker'            : t,
            'tipo_ativo'        : v['tipo_ativo'],
            'quantidade'        : v['qty'],
            'preco_medio'       : round(v['avg'], 6),
            'custo_total'       : round(abs(v['qty']) * v['avg'], 2),
            'data_ultima_compra': v['ultima_compra'],
        }
        for t, v in livro.items()
        if v['qty'] != 0
    ]
    df_custodia = (
        pd.DataFrame(registros)
        if registros
        else pd.DataFrame(columns=[
            'ticker', 'tipo_ativo', 'quantidade',
            'preco_medio', 'custo_total', 'data_ultima_compra',
        ])
    )

    return df, df_custodia


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — RESULTADO MENSAL + IR ESTIMADO
# ════════════════════════════════════════════════════════════════════════════

def _resultado_mensal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega P&L por mês, ticker e tipo de trade; calcula IR estimado.

    Regras fiscais
    --------------
    Day Trade (ações e futuros): 20% IR + 1% IRRF retido na fonte.
    Swing Trade ações          : 15% IR; isento se vendas mensais < R$ 20 k.
    Swing Trade futuros        : 15% IR; SEM isenção de R$ 20 k.
    Prejuízo                   : imposto_base = 0 (compensação interperíodo
                                 não implementada aqui).
    """
    df = df.copy()
    df['mes'] = df['data_pregao'].dt.to_period('M').astype(str)

    # P&L realizado:
    #   ações   → somente vendas (cv == 'V'); compras têm resultado_bruto = 0
    #   futuros → todas as linhas carregam resultado_bruto no valor_total ajustado
    mask_realizado = (df['tipo_ativo'] == 'FUTURO') | (df['cv'] == 'V')
    realizados = df[mask_realizado].copy()

    realizados['receita_linha'] = np.where(
        realizados['tipo_ativo'] == 'ACAO',
        realizados['preco_ajustado'] * realizados['quantidade'],
        realizados['resultado_bruto'].abs(),
    )

    # Receita de swing de ações por mês (base da isenção de R$ 20 k)
    receita_swing_acoes = (
        realizados[
            (realizados['tipo_trade'] == 'SWING_TRADE') &
            (realizados['tipo_ativo'] == 'ACAO') &
            (realizados['cv'] == 'V')
        ]
        .groupby('mes')['receita_linha']
        .sum()
        .rename('receita_swing_acoes_mes')
    )

    # IRRF por operação: 1% sobre cada dia POSITIVO individualmente
    # (não sobre o resultado mensal agregado, que pode ser negativo)
    # IRRF: corretoras usam truncamento (floor), não arredondamento bancário.
    # Exemplo: 1% × 172,91 = 1,7291 → nota mostra 1,72 (floor), não 1,73 (round).
    realizados['irrf_op'] = np.where(
        (realizados['tipo_trade'] == 'DAY_TRADE') & (realizados['resultado_bruto'] > 0),
        np.floor(realizados['resultado_bruto'] * IRRF_DAYTRADE * 100) / 100,
        0.0,
    )

    agg = (
        realizados
        .groupby(['mes', 'tipo_trade', 'tipo_ativo', 'ticker'])
        .agg(
            qtd_ops         = ('quantidade',      'count'),
            receita_bruta   = ('receita_linha',   'sum'),
            resultado_bruto = ('resultado_bruto', 'sum'),
            irrf_fonte      = ('irrf_op',         'sum'),   # soma dos dias lucrativos
        )
        .reset_index()
    )

    agg = agg.merge(receita_swing_acoes, on='mes', how='left')
    agg['receita_swing_acoes_mes'] = agg['receita_swing_acoes_mes'].fillna(0.0)

    # Isenção: Swing Trade ações com total mensal de vendas < R$ 20 k
    agg['isento'] = (
        (agg['tipo_trade'] == 'SWING_TRADE') &
        (agg['tipo_ativo'] == 'ACAO') &
        (agg['resultado_bruto'] > 0) &
        (agg['receita_swing_acoes_mes'] < LIMITE_ISENCAO_SWING)
    )

    agg['imposto_base'] = agg['resultado_bruto'].clip(lower=0.0)
    agg.loc[agg['isento'], 'imposto_base'] = 0.0

    agg['aliquota_ir'] = np.where(
        agg['tipo_trade'] == 'DAY_TRADE', ALIQUOTA_DAYTRADE, ALIQUOTA_SWING
    )
    agg.loc[agg['isento'], 'aliquota_ir'] = 0.0

    # irrf_fonte já calculado no groupby (soma de 1% de cada dia lucrativo)
    agg['irrf_fonte'] = agg['irrf_fonte'].round(2)
    agg['ir_devido'] = (
        (agg['imposto_base'] * agg['aliquota_ir'] - agg['irrf_fonte'])
        .clip(lower=0.0)
        .round(2)
    )

    for col in ['receita_bruta', 'resultado_bruto', 'imposto_base']:
        agg[col] = agg[col].round(2)

    return agg.drop(columns=['receita_swing_acoes_mes'])


# ════════════════════════════════════════════════════════════════════════════
#  FUNÇÃO PÚBLICA
# ════════════════════════════════════════════════════════════════════════════

def consolidar_notas(notas: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Consolida múltiplas notas em três DataFrames analíticos.

    Parâmetros
    ----------
    notas : list[dict]
        Cada elemento é o retorno de `extrair_dados_nota(texto_pdf)`.

    Retorna
    -------
    dict com:
        'operacoes'        — todas as ops (após row splitting) + PM + resultado
        'custodia'         — posições abertas com preço médio
        'resultado_mensal' — P&L por mês / tipo trade + IR estimado
    """
    if not notas:
        raise ValueError("Lista de notas vazia.")

    df_ops = pd.concat(
        [ratear_taxas(n) for n in notas], ignore_index=True
    )
    df_ops = _classificar_trade_type(df_ops)
    df_ops, df_custodia = _calcular_preco_medio_e_resultado(df_ops)
    df_resultado = _resultado_mensal(df_ops)

    return {
        'operacoes'        : df_ops,
        'custodia'         : df_custodia,
        'resultado_mensal' : df_resultado,
    }


# ════════════════════════════════════════════════════════════════════════════
#  DADOS DE TESTE
# ════════════════════════════════════════════════════════════════════════════

# Nota 1 — WIN mini-índice (futuro, day trade completo)
NOTA_WIN = {
    'data_pregao': '29/05/2026',
    'operacoes': [
        {'cv': 'C', 'ticker': 'WIN', 'quantidade': 1,
         'preco_unitario': 173990.00, 'valor_total':  236.20, 'dc': 'C'},
        {'cv': 'C', 'ticker': 'WIN', 'quantidade': 1,
         'preco_unitario': 174910.00, 'valor_total':   52.20, 'dc': 'C'},
        {'cv': 'V', 'ticker': 'WIN', 'quantidade': 1,
         'preco_unitario': 175130.00, 'valor_total':   -8.20, 'dc': 'D'},
        {'cv': 'V', 'ticker': 'WIN', 'quantidade': 1,
         'preco_unitario': 175220.00, 'valor_total':    9.80, 'dc': 'C'},
    ],
    'total_taxas': 1.00,
}

# Nota 2 — Compra de ações swing (base para PM)
NOTA_COMPRA_ACOES = {
    'data_pregao': '02/06/2026',
    'operacoes': [
        {'cv': 'C', 'ticker': 'PETR4', 'quantidade': 100,
         'preco_unitario': 35.50, 'valor_total': -3550.00, 'dc': 'D'},
        {'cv': 'C', 'ticker': 'VALE3', 'quantidade':  50,
         'preco_unitario': 65.00, 'valor_total': -3250.00, 'dc': 'D'},
    ],
    'total_taxas': 5.00,
}

# Nota 3 — Venda swing PETR4 + Day Trade completo ITUB4
NOTA_VENDA_MISTA = {
    'data_pregao': '15/06/2026',
    'operacoes': [
        {'cv': 'V', 'ticker': 'PETR4', 'quantidade':  50,
         'preco_unitario': 38.00, 'valor_total':  1900.00, 'dc': 'C'},
        {'cv': 'C', 'ticker': 'ITUB4', 'quantidade': 200,
         'preco_unitario': 28.00, 'valor_total': -5600.00, 'dc': 'D'},
        {'cv': 'V', 'ticker': 'ITUB4', 'quantidade': 200,
         'preco_unitario': 28.30, 'valor_total':  5660.00, 'dc': 'C'},
    ],
    'total_taxas': 8.00,
}

# Nota 4 — Aporte adicional PETR4 swing (rebaixa PM)
NOTA_APORTE = {
    'data_pregao': '20/06/2026',
    'operacoes': [
        {'cv': 'C', 'ticker': 'PETR4', 'quantidade': 200,
         'preco_unitario': 34.00, 'valor_total': -6800.00, 'dc': 'D'},
    ],
    'total_taxas': 3.50,
}

# Nota 5 — Day Trade PARCIAL (caso central do Row Splitting)
# C 100 BBAS3 @ 50,00  +  V 30 BBAS3 @ 52,00
# → dt_qty = min(100, 30) = 30
# → Row split: DT buy (30) + SW buy (70) + DT sell (30)
# → PM histórico = só SW buy (70 @ 50,00) → PM = 50,00
# → resultado DT = (52,00 − 50,00) × 30 = R$ 60,00
# Taxas = 0 para cálculo determinístico
NOTA_PARTIAL_DT = {
    'data_pregao': '01/07/2026',
    'operacoes': [
        {'cv': 'C', 'ticker': 'BBAS3', 'quantidade': 100,
         'preco_unitario': 50.00, 'valor_total': -5000.00, 'dc': 'D'},
        {'cv': 'V', 'ticker': 'BBAS3', 'quantidade':  30,
         'preco_unitario': 52.00, 'valor_total':  1560.00, 'dc': 'C'},
    ],
    'total_taxas': 0.00,
}


# ════════════════════════════════════════════════════════════════════════════
#  RUNNER DE TESTES
# ════════════════════════════════════════════════════════════════════════════

def _sep(titulo: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {titulo}")
    print(f"{'─' * 60}")


if __name__ == '__main__':
    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)

    notas = [NOTA_WIN, NOTA_COMPRA_ACOES, NOTA_VENDA_MISTA,
             NOTA_APORTE, NOTA_PARTIAL_DT]
    res   = consolidar_notas(notas)
    ops   = res['operacoes']
    cust  = res['custodia']
    rm    = res['resultado_mensal']

    # ── Exibição ────────────────────────────────────────────────────────────
    _sep('OPERAÇÕES (após row splitting)')
    cols = ['data_pregao', 'ticker', 'tipo_ativo', 'tipo_trade',
            'cv', 'quantidade', 'preco_ajustado', 'resultado_bruto']
    print(ops[cols].to_string(index=False))

    _sep('CUSTÓDIA / PREÇO MÉDIO')
    print(cust.to_string(index=False) if not cust.empty else '  (vazio)')

    _sep('RESULTADO MENSAL')
    cols_rm = ['mes', 'tipo_trade', 'tipo_ativo', 'ticker',
               'resultado_bruto', 'isento', 'aliquota_ir', 'ir_devido']
    print(rm[cols_rm].to_string(index=False))

    # ════════════════════════════════════════════════════════════════════════
    #  ASSERTIONS
    # ════════════════════════════════════════════════════════════════════════
    _sep('VALIDAÇÕES')

    # ── A. BM&F WIN — futuros DT ────────────────────────────────────────────
    win_ops = ops[ops['ticker'] == 'WIN']
    assert len(win_ops) == 4,                    '❌ WIN: esperado 4 ops'
    assert (win_ops['tipo_trade'] == 'DAY_TRADE').all(), '❌ WIN: todas DT'
    pl_win = win_ops['resultado_bruto'].sum()
    assert abs(pl_win - 289.00) < 0.02,         f'❌ WIN P&L esperado 289,00 | obtido {pl_win:.2f}'
    print(f'✅ WIN  P&L = R$ {pl_win:.2f}  (esperado 289,00)')

    # ── B. ITUB4 — Day Trade completo: resultado isolado pelos preços do dia ─
    # resultado_correto  = (pm_v_adj − pm_c_adj) × 200  ≈ 53
    # resultado_incorreto = (pm_v_adj − 0) × 200        ≈ 5656  (bug corrigido)
    itub_ops     = ops[ops['ticker'] == 'ITUB4']
    total_vol_n3 = 1900 + 5600 + 5660
    taxa_itub_c  = 8.00 * 5600 / total_vol_n3
    taxa_itub_v  = 8.00 * 5660 / total_vol_n3
    pa_itub_c    = 28.00 + taxa_itub_c / 200
    pa_itub_v    = 28.30 - taxa_itub_v / 200
    resultado_dt_esperado = (pa_itub_v - pa_itub_c) * 200

    resultado_itub = itub_ops['resultado_bruto'].sum()
    assert abs(resultado_itub - resultado_dt_esperado) < 0.01, \
        f'❌ ITUB4 DT resultado: esperado ≈{resultado_dt_esperado:.2f} | obtido {resultado_itub:.2f}'
    assert resultado_itub < 200, \
        f'❌ ITUB4 usando PM=0 (bug antigo): {resultado_itub:.2f}'
    print(f'✅ ITUB4 DT resultado = R$ {resultado_itub:.2f}  '
          f'(intraday; isolado do PM histórico)')

    # ── C. ITUB4 — DT NÃO aparece na custódia ───────────────────────────────
    assert 'ITUB4' not in cust['ticker'].values, \
        '❌ ITUB4 não deve aparecer na custódia (DT completo)'
    print('✅ ITUB4 ausente da custódia (Day Trade sem posição residual)')

    # ── D. Row Splitting — BBAS3 partial DT ─────────────────────────────────
    bbas_ops = ops[ops['ticker'] == 'BBAS3']
    # Após split: 3 linhas físicas (DT buy 30, SW buy 70, DT sell 30)
    assert len(bbas_ops) == 3, \
        f'❌ BBAS3: esperado 3 linhas após split | obtido {len(bbas_ops)}'

    bbas_dt = bbas_ops[bbas_ops['tipo_trade'] == 'DAY_TRADE']
    bbas_sw = bbas_ops[bbas_ops['tipo_trade'] == 'SWING_TRADE']

    assert bbas_dt[bbas_dt['cv'] == 'C']['quantidade'].iloc[0] == 30, \
        '❌ BBAS3 DT buy qty esperado 30'
    assert bbas_dt[bbas_dt['cv'] == 'V']['quantidade'].iloc[0] == 30, \
        '❌ BBAS3 DT sell qty esperado 30'
    assert bbas_sw['quantidade'].iloc[0] == 70, \
        '❌ BBAS3 SW buy qty esperado 70'
    print('✅ BBAS3 row split: DT-buy=30 | DT-sell=30 | SW-buy=70')

    # ── E. BBAS3 DT — resultado calculado pelos preços intraday ─────────────
    resultado_bbas3_dt = bbas_dt[bbas_dt['cv'] == 'V']['resultado_bruto'].iloc[0]
    assert abs(resultado_bbas3_dt - 60.00) < 0.01, \
        f'❌ BBAS3 DT resultado esperado 60,00 | obtido {resultado_bbas3_dt:.2f}'
    print(f'✅ BBAS3 DT resultado = R$ {resultado_bbas3_dt:.2f}  '
          f'(52,00 − 50,00) × 30')

    # ── F. BBAS3 DT — NÃO altera o preço médio histórico ───────────────────
    pm_bbas3 = cust.loc[cust['ticker'] == 'BBAS3', 'preco_medio'].iloc[0]
    qty_bbas3 = cust.loc[cust['ticker'] == 'BBAS3', 'quantidade'].iloc[0]
    assert abs(pm_bbas3 - 50.00) < 0.001, \
        f'❌ BBAS3 PM esperado 50,00 (só SW buy) | obtido {pm_bbas3:.4f}'
    assert qty_bbas3 == 70, \
        f'❌ BBAS3 custódia: esperado 70 | obtido {qty_bbas3}'
    print(f'✅ BBAS3 custódia: qty={qty_bbas3} PM=R${pm_bbas3:.2f}  '
          f'(DT não alterou PM histórico)')

    # ── G. Preço médio PETR4 após notas 2, 3 e 4 ───────────────────────────
    taxa_petr4_n2 = 5.00 * (3550 / (3550 + 3250))
    pm_petr4_n2   = 35.50 + taxa_petr4_n2 / 100
    preco_aj_n4   = 34.00 + 3.50 / 200
    # Após V 50 (swing), resta 50 unid; depois C 200 → PM ponderado
    pm_esperado   = (50 * pm_petr4_n2 + 200 * preco_aj_n4) / 250
    pm_obtido     = cust.loc[cust['ticker'] == 'PETR4', 'preco_medio'].iloc[0]
    assert abs(pm_obtido - pm_esperado) < 0.001, \
        f'❌ PETR4 PM esperado {pm_esperado:.4f} | obtido {pm_obtido:.4f}'
    print(f'✅ PETR4 PM = R$ {pm_obtido:.4f}  (esperado {pm_esperado:.4f})')

    # ── H. PETR4 swing — isento (vendas < R$ 20 k) ─────────────────────────
    row_sw = rm[
        (rm['ticker'] == 'PETR4') & (rm['tipo_trade'] == 'SWING_TRADE')
    ]
    assert not row_sw.empty and row_sw['isento'].iloc[0], \
        '❌ PETR4 swing deve ser isento (vendas < R$ 20 k)'
    print('✅ PETR4 Swing Trade ISENTO (vendas mensais < R$ 20 000)')

    print(f'\n{"═" * 60}')
    print('  ✅  Todos os asserts passaram!')
    print(f'{"═" * 60}\n')
