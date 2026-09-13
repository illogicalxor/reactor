# Consenso e surpresa: guia de uso real

O classificador de texto mede **stance** (dovish → bullish, hawkish → bearish
para US equities). Impacto real é **stance menos o que já estava precificado**.
Sem consenso explícito, o pipeline não sabe a surpresa e **não deve gerar
sinal acionável**.

## Inferência

Informe o consenso em `/v1/analyze-video` (form fields) ou no JSON inicial do
WebSocket:

- `consensus_priced_bullish` (0–1): o que o mercado precificava antes do evento.
  Ex.: 0.8 = mercado já precificava corte/alívio; 0.2 = precificava aperto.
  Fonte típica: Fed Funds futures, OIS, opções, survey median.
- `consensus_confidence` (0–1): confiança no consenso.
- `consensus_source`: ex. `fed_funds_futures`, `ois`, `options`, `survey`.
- `consensus_horizon_seconds`: horizonte do impacto (padrão 300 = 5 min).
- `consensus_expected_move_bps`: movimento esperado/ruído (alarga a margem).

Exemplo:

```bash
curl -X POST http://localhost:8000/v1/analyze-video \
  -F file=@fomc.mp4 \
  -F 'asset_context=S&P 500' \
  -F 'consensus_priced_bullish=0.7' \
  -F 'consensus_confidence=0.85' \
  -F 'consensus_source=fed_funds_futures' \
  -F 'consensus_expected_move_bps=8'
```

Resposta passa a incluir:

- `consensus_mode`: `none | explicit | proxy`
- `priced_bullish_probability`, `surprise_score`, `surprise_confidence`
- `actionable`: só `true` quando há surpresa suficiente e calibrada
- `low_signal_reason`: `consensus_unknown | consensus_priced_in |
  weak_surprise | information_effect_risk | weak_stance | no_directional_content`
- `neutral_margin_applied`, `information_effect_risk`

Regras:

1. Sem consenso (`mode=none`): probabilidades encolhem para 50/50,
   `actionable=false`, `low_signal=true`, `reason=consensus_unknown`.
   Downstream deve **abster-se**.
2. Surpresa pequena (`|stance − priced| < POLICY_SIGNAL_MIN_SURPRISE`, padrão
   0.15): `consensus_priced_in`, não acionável. O mercado já esperava.
3. Linguagem dovish + risco (recessão/stress) com viés bullish:
   `information_effect_risk`, nunca acionável como bullish.
4. `actionable=true` exige stance forte **e** surpresa suficiente.

Desligar `POLICY_SIGNAL_REQUIRE_CONSENSUS_FOR_ACTION` só para ablação.

## Treino (outra máquina, GPU)

Stance ≠ rótulo de impacto. Gere rótulos de impacto com:

```bash
policy-signal-label-impact \
  --input data/impact_canonical.example.csv \
  --output data/impact_train.csv \
  --manifest data/impact_manifest.json \
  --band-multiplier 1.0 --band-floor-bps 5.0 --min-surprise-bps 2.0
```

O CSV canônico exige `event_id,text,asset_context,price_before,price_after,
horizon_seconds` + opcionais `spread_bps,expected_move_bps,realized_vol_bps,
transaction_cost_bps,confounded,stale_quote,illiquid,uncertain_timing,
surprise_bps`.

- Banda neutra calibrada por microestrutura, não 10 bps fixos.
- `confounded/stale/illiquid/uncertain_timing` → excluído, nunca treina.
- `surprise_bps` abaixo do mínimo → neutral.
- Peso `weight` ∝ distância fora da banda.

Treine o binário com o CSV de impacto (mesmo `policy-signal-train`), split por
evento/data, um dataset por ativo × horizonte. Nunca misture 1/5/30 min no
mesmo rótulo.

## O que ainda falta para trading real

- Feed vivo de consenso (futuros/OIS) com `as_of` antes do evento.
- Teste fora do tempo maior, paper-trading com spread/custos/latência.
- Calibração por regime e monitoramento de drift.
- Licenças comerciais (atuais são CC BY-NC).
