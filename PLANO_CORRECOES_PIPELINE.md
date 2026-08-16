# Plano de correções do pipeline

## Objetivo

Corrigir os comportamentos observados na coletiva do FOMC de 29 de julho de 2026
sem ajustar o sistema apenas para um evento. O alvo é produzir um sinal atribuível à
autoridade econômica, rejeitar conteúdo sem informação de política e medir impacto
contra uma janela de mercado explicitamente definida.

## Diagnóstico reproduzido

O pipeline atual apresenta quatro problemas independentes:

1. O Whisper fornece texto e timestamps, mas não identifica o locutor. Perguntas de
   jornalistas entram no mesmo fluxo das respostas do Chair.
2. O v2 foi treinado somente com exemplos `hawkish` e `dovish`. Os exemplos
   `neutral` e `irrelevant` foram excluídos, então o classificador é obrigado a dar
   direção a saudações, comentários processuais e perguntas.
3. O rótulo de treino é stance monetário convertido por regra, não retorno observado
   do ativo. Isso não aprende surpresa em relação ao que o mercado já precificava.
4. A mesma meia-vida de 90 segundos usada no streaming era aplicada ao vídeo
   completo. Os comentários finais, mesmo pouco informativos, dominavam 45 minutos
   de coletiva.

A ablação no vídeo real produziu:

| Agregação | Probabilidade bullish | Resultado |
|---|---:|---|
| Recência de 90 s | 55,51% | bullish, low signal |
| Recência de 15 min | 49,52% | bearish, low signal |
| Sem decaimento temporal | 45,28% | bearish, low signal |
| Somente trechos com conteúdo monetário | 39,24% | bearish |
| Conteúdo monetário sem perguntas aparentes | 36,93% | bearish |

Isso mostra que diarização é necessária, mas não suficiente. Relevância e agregação
também precisam ser corrigidas.

## Arquitetura alvo

```text
vídeo
  -> áudio normalizado + frames sincronizados
  -> ASR com timestamps por palavra
  -> diarização: quem falou quando
  -> identificação de papel: policy_maker | journalist | unknown
  -> construção de turnos pergunta/resposta
  -> gate interno: irrelevant | neutral | directional
  -> classificador de stance e classificador de impacto/surpresa
  -> agregação somente de falas direcionais da autoridade
  -> bullish | bearish | low_signal com evidências e cobertura
```

## Fase 1: agregação offline e streaming

Implementado neste ciclo:

- gravações completas não usam decaimento temporal por padrão;
- o rolling signal do WebSocket conserva a meia-vida de 90 segundos;
- `POLICY_SIGNAL_OFFLINE_RECENCY_HALF_LIFE_SECONDS` permite reativar e calibrar
  decaimento offline explicitamente;
- há um teste de regressão garantindo que uma fala final arbitrária não receba
  preferência apenas por estar no fim;
- perguntas e respostas recebem IDs ligados por `question_turn_id`;
- a API preserva uma série `turns` com um sinal para cada fala da autoridade,
  incluindo respostas neutras ou irrelevantes como 50/50 e `low_signal=true`.

O rótulo único do evento agora é uma visão adicional e não substitui a série por
resposta. Cada conexão WebSocket mantém estado isolado de pergunta, resposta,
locutor textual e contadores de turno, inclusive quando uma pergunta atravessa
janelas. A associação acústica de clusters entre janelas continua dependente de
voiceprint, pois os IDs anônimos do pyannote são locais a cada execução.

## Fase 2: separar Chair e jornalistas

Implementado neste ciclo:

- metadados `speaker_id`, `speaker_role`, `role_confidence`, `role_source` e
  `question_context` percorrem ASR, classificação e resposta da API;
- `turn_id` e `question_turn_id` ligam cada evidência à resposta e à pergunta que a
  originou;
- o fallback textual separa introduções explícitas de jornalistas e conserva a
  pergunta como contexto da resposta;
- o backend opcional `pyannote` faz diarização no áudio, usa a timeline exclusiva
  para casar clusters aos timestamps por palavra do ASR, divide segmentos mistos
  nos limites de troca de voz e ancora como autoridade o locutor dominante da
  abertura;
- falas de jornalistas e clusters desconhecidos não entram no classificador;
- testes garantem que segmentos não sejam unidos através de uma troca de locutor.

Pendente: voiceprint/rosto cadastrado para confirmar que o cluster ancorado é mesmo
o Chair e avaliação de DER no vídeo real. O estado textual de pergunta, resposta e
papel já persiste por conexão WebSocket; clusters acústicos ainda são locais a cada
janela do pyannote.

### Caminho de áudio

Usar WhisperX para alinhamento por palavra e `pyannote.audio` Community-1 para
diarização local. Diarização gera clusters como `SPEAKER_00`; ela não garante o nome
real do locutor.

Para coletivas do FOMC, mapear o Chair por evidências combinadas:

1. o locutor do bloco inicial contínuo é o candidato principal;
2. comparar o embedding de voz com uma amostra limpa previamente cadastrada;
3. confirmar com presença facial e locutor ativo no vídeo;
4. usar `unknown` quando as evidências discordarem.

Uma alternativa comercial do pyannote oferece voiceprints e identificação pronta,
mas deve ser avaliada separadamente por privacidade, custo e dependência externa.

### Caminho de vídeo

A filmagem analisada alterna enquadramentos nítidos do Chair e dos jornalistas. Um
detector audiovisual de locutor ativo, como TalkNet, pode associar fala ao rosto que
está falando e resolver trocas de câmera ou sobreposição. Reconhecimento facial deve
identificar o Chair cadastrado; os demais rostos podem permanecer `journalist` sem
identidade nominal.

Perguntas não devem ser descartadas totalmente. Elas entram como contexto:

```text
[QUESTION] <texto do jornalista>
[POLICY_MAKER_ANSWER] <resposta da autoridade>
```

Somente a resposta gera evidência direcional. Isso preserva o assunto e evita tratar
uma pergunta sobre alta de juros como posição do Chair.

Campos necessários no domínio:

- `speaker_id`;
- `speaker_role`;
- `speaker_confidence`;
- `face_track_id`;
- `active_speaker_confidence`;
- `turn_id`;
- `question_turn_id`;
- `role_source` (`audio`, `video`, `voiceprint`, `combined`).

## Fase 3: neutralidade e relevância

Implementado neste ciclo:

- gate `irrelevant | neutral | directional` antes do v2-large;
- apenas conteúdo direcional da autoridade é classificado;
- ausência de conteúdo direcional retorna 50/50 com `low_signal=true`;
- a API expõe cobertura e contagens por papel/relevância para auditoria;
- backends podem ser desabilitados por configuração para ablações.

O baseline heurístico foi substituído opcionalmente pelo checkpoint
`policy-relevance-gate-v1`, treinado em CUDA com as três classes e split cronológico.
No teste de 2023–2024, o F1 macro foi 0,6734, contra 0,3692 do heurístico nos mesmos
59 exemplos. O gate neural está integrado pelo backend `fine-tuned`; o heurístico
permanece disponível para rollback e instalações sem os pesos locais.

O conjunto de regressão local agora cobre 28 falas `irrelevant`, oito `neutral` e
oito `directional`. Nele, o checkpoint mais o guardrail processual obteve F1 macro
0,7909 e recall 1,0 de `irrelevant`; o resultado bruto e o pós-processado foram
versionados separadamente. O próximo treino aceita exemplos humanos suplementares
somente no split de treino e grava calibração por temperature scaling usando a
validação. O checkpoint v1 atual é anterior a esse calibrador e usa temperatura 1,0.

Pendente: repetir o treino na máquina com GPU usando exemplos humanos de múltiplos
eventos, medir calibração fora da amostra e validar a separação de papéis junto com
diarização real. O conjunto local é sintético e serve como regressão, não como
benchmark independente de produção.

O gate interno usa os rótulos originalmente disponíveis:

- `irrelevant`: apresentação, agradecimento, logística e comentários sobre imprensa;
- `neutral`: conteúdo econômico sem direção suficiente;
- `directional`: conteúdo que segue para o classificador bullish/bearish.

O contrato externo pode continuar binário. Se não houver massa direcional suficiente,
o serviço retorna o rótulo contratual com `low_signal=true` e informa cobertura baixa.

O filtro definitivo não depende apenas de palavras-chave. O heurístico usado na
ablação continua útil como baseline, mas perde negações, sinônimos e respostas cujo
significado depende da pergunta.

## Fase 4: impacto de mercado e surpresa

Separar duas tarefas:

1. `stance`: hawkish, neutral ou dovish;
2. `asset impact`: retorno esperado para um ativo e horizonte específicos, condicionado
   às expectativas anteriores ao evento.

Treinar o segundo estágio com retornos intradiários observados. Para cada resposta,
guardar preço anterior, preço posterior, horizonte, volatilidade, spread e eventos
concorrentes. SPY ou ES podem representar ações dos EUA durante seus respectivos
horários líquidos.

Avaliar horizontes de 1, 5 e 30 minutos separadamente. Não misturar reação ao
comunicado das 14h com reação à coletiva das 14h30.

## Fase 5: expressões faciais e prosódia

Expressão facial deve ser uma feature auxiliar, nunca um rótulo direto de emoção.
Movimentos faciais não permitem inferir de forma universal e confiável estados como
medo, confiança ou sinceridade.

Features defensáveis para experimentação:

- blendshapes ou action units normalizados contra o baseline do próprio orador;
- direção do olhar e frequência de consulta a notas;
- piscadas, movimentos de cabeça e intensidade gestual;
- pausas, velocidade, pitch e intensidade vocal;
- divergência entre texto, voz e movimento facial.

Usar apenas deltas dentro do mesmo evento/orador, com timestamps sincronizados ao
mercado. Promover essas features somente se uma ablação fora do tempo melhorar
calibração e retorno após custos. MediaPipe Face Landmarker pode extrair geometria e
blendshapes; TalkNet é mais apropriado para identificar o rosto que está falando.

## Avaliação e gates de promoção

Criar conjuntos separados por evento e data, com transcrições oficiais usadas como
referência, nunca como entrada disponível antes da publicação.

Métricas mínimas:

- WER do ASR e erro de fronteira de turnos;
- diarization error rate;
- precisão e recall de `policy_maker`, priorizando alta precisão;
- F1 de `irrelevant`, `neutral` e `directional`;
- cobertura e acurácia após abstinência;
- calibração por orador, tipo de evento e horizonte;
- acerto direcional e retorno líquido em paper trading fora do tempo.

A coletiva de julho de 2026 deve virar teste de regressão, mas não exemplo de treino:

- o sistema não pode atribuir perguntas ao Chair;
- comentários finais processuais não podem dominar o evento;
- para a janela da coletiva em ações dos EUA, o resultado aceitável é bearish ou
  abstinência, nunca um bullish acionável.

## Ordem recomendada

1. concluir regressões da agregação offline;
2. treinar o gate `irrelevant/neutral/directional`;
3. integrar diarização e papel do locutor;
4. construir dataset com retornos e expectativas;
5. testar prosódia;
6. testar features faciais somente após os anteriores.

Referências técnicas:

- WhisperX: https://github.com/m-bain/whisperX
- pyannote.audio: https://github.com/pyannote/pyannote-audio
- TalkNet active-speaker detection: https://github.com/TaoRuijie/TalkNet-ASD
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Revisão sobre inferência de emoção facial: https://doi.org/10.1177/1529100619832930
- Evidência multimodal em comunicação de bancos centrais:
  https://doi.org/10.1016/j.jmoneco.2023.09.002
