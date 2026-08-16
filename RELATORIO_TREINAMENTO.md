# Relatório de treinamento do Policy Signal

Data da execução: 15 de agosto de 2026  
Escopo: classificação de linguagem de autoridades econômicas dos Estados Unidos em
`bullish` ou `bearish` para o contexto `US equities`.

## 1. Resumo executivo

Foram treinadas duas versões do classificador de texto:

1. `policy-signal-baseline-v1`, criado a partir do `FacebookAI/roberta-base` com
   486 exemplos de treino;
2. `policy-signal-baseline-v2-large`, criado a partir do v1 com uma base ampliada
   para 1.712 exemplos.

O v2 melhorou sobre o v1 tanto na validação quanto no teste fora do tempo. No teste
de 2023–2024, a acurácia passou de 0,675 para 0,725 e o F1 macro passou de 0,675
para 0,725. A configuração local em `.env` aponta para o v2, enquanto o v1 foi
preservado como opção de rollback.

O resultado é um baseline funcional para pesquisa e demonstrações. Ele não é um
modelo validado para trading ou uso comercial.

## 2. O que foi treinado

O projeto possui dois componentes de ML independentes:

- `faster-whisper`, responsável por transformar áudio em texto;
- o classificador RoBERTa, responsável por classificar cada trecho transcrito.

Somente o classificador de texto foi treinado. O Whisper permaneceu congelado com
pesos pré-treinados, conforme definido na documentação do projeto. Consequentemente,
o treino usou frases e rótulos, não vídeos.

O texto entregue ao classificador segue o formato usado pelo serviço:

```text
[ASSET] US equities [STATEMENT] <texto da fala>
```

O mapeamento de rótulos do checkpoint é:

```text
bearish = 0
bullish = 1
```

## 3. Ambiente de execução

- Sistema: Linux x86-64;
- Python: 3.12.14, em `.venv`;
- GPU: NVIDIA GeForce RTX 3050 com 8 GB de VRAM;
- driver NVIDIA: 610.57.04;
- PyTorch: 2.12.1 com CUDA 13.0;
- Transformers: 4.57.6;
- Datasets: 3.6.0;
- scikit-learn: 1.9.0.

A GPU foi validada antes do treino com uma operação real de tensor. O backend de
inferência também foi carregado posteriormente em `cuda:0`.

## 4. Dados utilizados

### 4.1. Primeira versão

O v1 usou o dataset
[`gtfintechlab/federal_reserve_system`](https://huggingface.co/datasets/gtfintechlab/federal_reserve_system),
que contém 1.000 frases anotadas. Foram mantidos apenas os rótulos direcionais:

- `dovish` foi mapeado para `bullish`;
- `hawkish` foi mapeado para `bearish`;
- `neutral` e `irrelevant` foram excluídos.

Após o filtro, restaram 631 observações, separadas cronologicamente:

| Split | Período | Exemplos |
|---|---:|---:|
| Treino | até 2018 | 486 |
| Validação | 2019–2022 | 105 |
| Teste | 2023–2024 | 40 |

Licença da fonte: CC BY-NC-SA 4.0.

### 4.2. Base ampliada

O v2 adicionou o dataset
[`gtfintechlab/fomc_communication`](https://huggingface.co/datasets/gtfintechlab/fomc_communication),
que possui 2.480 frases publicadas. Dessas, 1.256 tinham rótulo `hawkish` ou
`dovish`; as neutras foram descartadas para preservar o contrato binário.

O split publicado por essa fonte é aleatório, e alguns valores de ano não são
consistentes com o conteúdo das frases. Por isso, esses exemplos foram usados apenas
no treino. Os conjuntos cronológicos anteriores foram mantidos para validação e
teste.

Composição final do treino ampliado:

| Fonte | Exemplos |
|---|---:|
| `federal_reserve_system` | 486 |
| `fomc_communication` | 1.214 |
| `data/example.csv` | 12 |
| **Total** | **1.712** |

Distribuição de classes:

| Classe | Exemplos |
|---|---:|
| `bearish` | 801 |
| `bullish` | 911 |

Licença do `fomc_communication`: CC BY-NC 4.0.

### 4.3. Limpeza e prevenção de vazamento

Foi aplicado o seguinte processo:

1. normalização Unicode NFKC;
2. conversão para minúsculas com `casefold`;
3. substituição de sequências de caracteres não alfanuméricos;
4. remoção de duplicatas textuais;
5. exclusão de um texto que possuía rótulos conflitantes na nova fonte;
6. remoção de sobreposições textuais exatas com validação e teste;
7. preservação integral dos splits cronológicos de avaliação.

Os CSVs gerados continuam ignorados pelo Git porque são dados derivados. Seus hashes
SHA-256 e a proveniência estão registrados nos manifests de cada modelo.

## 5. Configuração dos treinamentos

### 5.1. v1

| Parâmetro | Valor |
|---|---:|
| Base | `FacebookAI/roberta-base` |
| Épocas | 3 |
| Batch size | 8 |
| Learning rate | `2e-5` |
| Seed | 42 |
| Exemplos de treino | 486 |

### 5.2. v2-large

| Parâmetro | Valor |
|---|---:|
| Base | `models/policy-signal-baseline-v1` |
| Épocas | 3 |
| Batch size | 8 |
| Learning rate | `1e-5` |
| Seed | 42 |
| Exemplos de treino | 1.712 |
| Passos de otimização | 642 |
| Tempo de treino | 76,5 segundos |

O treinamento incremental do v2 seguiu a recomendação de manter o modelo anterior
imutável, usar a versão anterior como base e escrever a nova versão em outro
diretório.

## 6. Resultados

### 6.1. Validação 2019–2022

| Métrica | v1 | v2-large | Variação |
|---|---:|---:|---:|
| Acurácia | 0,7619 | 0,8857 | +0,1238 |
| F1 macro | 0,7166 | 0,8539 | +0,1373 |
| Loss | 0,4712 | 0,3150 | -0,1562 |

O melhor checkpoint do v2 foi selecionado automaticamente por F1 macro.

### 6.2. Teste fora do tempo, 2023–2024

| Métrica | v1 | v2-large |
|---|---:|---:|
| Acurácia | 0,6750 | 0,7250 |
| Balanced accuracy | 0,6867 | 0,7379 |
| F1 macro | 0,6748 | 0,7248 |
| Brier score (`bullish`) | 0,2374 | 0,2062 |
| ECE, 10 bins | 0,1511 | 0,1268 |
| Taxa `low_signal` | 0,0500 | 0,1000 |
| Acurácia após abstinência | 0,6842 | 0,7500 |

Matriz de confusão do v2:

| Real / Previsto | `bearish` | `bullish` |
|---|---:|---:|
| `bearish` | 15 | 8 |
| `bullish` | 3 | 14 |

Resultados por classe no v2:

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| `bearish` | 0,8333 | 0,6522 | 0,7317 | 23 |
| `bullish` | 0,6364 | 0,8235 | 0,7179 | 17 |

O conjunto de teste tem apenas 40 exemplos. Portanto, essas métricas possuem alta
incerteza e não devem ser tratadas como estimativa definitiva de desempenho.

### 6.3. Verificações adicionais

- 12 testes automatizados do projeto passaram;
- o checkpoint foi carregado pelo `FineTunedSignalClassifier` real;
- a inferência foi executada em `cuda:0`;
- o exemplo “higher for longer” foi classificado como `bearish` com confiança 0,984;
- 10 dos 12 exemplos de `data/example.csv` foram reproduzidos pelo v2.

O último item é apenas um teste de sanidade, não uma avaliação independente, pois
esses 12 exemplos foram incluídos no treino ampliado.

## 7. Pesos e artefatos

### Modelo atual

Diretório:

```text
models/policy-signal-baseline-v2-large/
```

Pesos finais:

```text
models/policy-signal-baseline-v2-large/model.safetensors
```

Tamanho dos pesos: 498.612.824 bytes, aproximadamente 499 MB.

SHA-256:

```text
bd56001c69ae2998d3f0a6ec3dd5fc2df643b58efcbca7388fb239ef42b010f9
```

Arquivos relevantes:

- [`training_manifest.json`](models/policy-signal-baseline-v2-large/training_manifest.json):
  hiperparâmetros, hashes e métricas de validação;
- [`dataset_manifest.json`](models/policy-signal-baseline-v2-large/dataset_manifest.json):
  fontes, licenças, contagens e deduplicação;
- [`test_evaluation.json`](models/policy-signal-baseline-v2-large/test_evaluation.json):
  métricas fora do tempo;
- [`repository_examples_evaluation.json`](models/policy-signal-baseline-v2-large/repository_examples_evaluation.json):
  teste de sanidade dos exemplos locais;
- tokenizer, vocabulário, merges e `config.json` necessários para inferência.

### Rollback

O modelo anterior permanece em:

```text
models/policy-signal-baseline-v1/
```

Trocar o valor de `POLICY_SIGNAL_CLASSIFIER_MODEL` permite voltar ao v1 sem
sobrescrever nenhum peso.

## 8. Uso local

A `.env` atual contém:

```dotenv
POLICY_SIGNAL_CLASSIFIER_BACKEND=fine-tuned
POLICY_SIGNAL_CLASSIFIER_MODEL=models/policy-signal-baseline-v2-large
POLICY_SIGNAL_WHISPER_DEVICE=cuda
POLICY_SIGNAL_WHISPER_COMPUTE_TYPE=float16
```

Inicialização:

```bash
. .venv/bin/activate
uvicorn policy_signal.api:app --host 0.0.0.0 --port 8000
```

Análise de entrevista em inglês:

```bash
curl -X POST http://localhost:8000/v1/analyze-video \
  -F file=@entrevista.mp4 \
  -F 'asset_context=S&P 500'
```

O Whisper `small.en` será baixado no primeiro uso caso ainda não esteja no cache.

## 9. Git e armazenamento dos pesos

A regra ampla `models/` foi removida do `.gitignore`. Os pesos finais, tokenizer,
configuração e manifests das duas versões agora aparecem para o Git.

Os diretórios `models/**/checkpoint-*/` continuam ignorados. Eles armazenam estados
retomáveis de treinamento, incluindo cópias dos pesos e arquivos de otimizador de
aproximadamente 997 MB cada. Esses arquivos não são necessários para inferência e
adicionariam vários gigabytes redundantes ao repositório.

O Git LFS não está instalado neste ambiente. Cada `model.safetensors` tem cerca de
499 MB e pode exceder o limite de arquivos do servidor Git. Antes de adicionar ou
enviar os pesos, recomenda-se instalar e configurar Git LFS:

```bash
git lfs install
git lfs track 'models/**/model.safetensors'
git add .gitattributes models/ RELATORIO_TREINAMENTO.md .gitignore
```

Nenhum `git add`, commit ou push foi executado durante este trabalho.

## 10. Limitações e critérios de uso

1. Os rótulos principais são uma supervisão fraca baseada em stance monetário.
   Eles não são retornos observados do S&P 500 ou de outro ativo.
2. Linguagem dovish costuma ser positiva para ações, mas pode ser negativa quando
   sinaliza recessão ou risco financeiro. Esse “information effect” não está
   resolvido pela base atual.
3. As licenças CC BY-NC e CC BY-NC-SA restringem uso comercial.
4. O teste possui somente 40 exemplos.
5. O modelo foi treinado para inglês, autoridades dos EUA e `US equities`. Outros
   ativos exigem rótulos próprios.
6. Não foram medidos WER do Whisper, latência completa de vídeo, drift por orador,
   paper trading ou custos de execução.
7. O modelo não deve ser tratado como previsão de preço ou recomendação financeira.

Ele já pode ser usado em entrevistas para demonstração, pesquisa e coleta de novos
exemplos, desde que respostas `low_signal` sejam tratadas como abstinência. Não deve
ser usado para colocar capital em risco.

## 11. Próximos passos recomendados

1. alinhar pronunciamentos oficiais com retornos intradiários de SPY ou ES;
2. definir horizonte e banda neutra, por exemplo 5 ou 30 minutos;
3. separar todos os trechos do mesmo evento no mesmo split;
4. criar um teste fora do tempo maior e imutável;
5. calibrar probabilidades e o limiar de `low_signal`;
6. medir WER e latência com entrevistas reais;
7. executar paper trading com spread, custos e atraso do sinal;
8. substituir ou relicenciar os datasets antes de qualquer uso comercial.

## 12. Gate de relevância treinado em GPU

Em 16 de agosto de 2026 foi treinado um terceiro checkpoint, independente do
classificador direcional v2-large. Sua tarefa interna é decidir entre `irrelevant`,
`neutral` e `directional`; apenas o último rótulo segue para a decisão
`bullish`/`bearish`.

O treino reutilizou o backbone `models/policy-signal-baseline-v2-large`, trocou a
cabeça por três classes e foi executado em uma NVIDIA GeForce RTX 3050 com FP16:

| Parâmetro | Valor |
|---|---:|
| Épocas concluídas | 4 |
| Melhor checkpoint | época 3 (`checkpoint-1182`) |
| Batch size | 8 |
| Learning rate | `1e-5` |
| Seed | 42 |
| Exemplos de treino | 3.151 |
| Tempo de treino | 131,8 segundos |

Os splits foram deduplicados e separados cronologicamente: anos até 2018 no treino,
2019–2022 na validação e 2023–2024 no teste. O corpus
`fomc_communication`, cujo split publicado é aleatório, entrou somente no treino.

| Split | `irrelevant` | `neutral` | `directional` | Total |
|---|---:|---:|---:|---:|
| Treino | 35 | 1.426 | 1.690 | 3.151 |
| Validação | 4 | 53 | 105 | 162 |
| Teste | 2 | 17 | 40 | 59 |

No teste fora do tempo, o modelo neural superou o gate heurístico:

| Métrica | Heurístico | Gate treinado | Variação |
|---|---:|---:|---:|
| Acurácia | 0,5254 | 0,7119 | +0,1864 |
| Balanced accuracy | 0,5975 | 0,6324 | +0,0348 |
| F1 macro | 0,3692 | 0,6734 | +0,3043 |

Matriz de confusão do gate treinado, na ordem `irrelevant`, `neutral`,
`directional`:

| Real / Previsto | `irrelevant` | `neutral` | `directional` |
|---|---:|---:|---:|
| `irrelevant` | 1 | 1 | 0 |
| `neutral` | 0 | 11 | 6 |
| `directional` | 0 | 10 | 30 |

O artefato final está em `models/policy-relevance-gate-v1`. Os pesos têm
498.615.900 bytes e SHA-256
`1261103c34149a29cf5b12bbb68fe6d97595418fe86f32288ca390df79b943be`.
O manifesto registra os hashes dos três splits, e a comparação com o heurístico foi
salva no próprio diretório do modelo.

A avaliação da classe `irrelevant` ainda é inconclusiva: há somente dois exemplos
dessa classe no teste cronológico. Portanto, o checkpoint corrige o principal
problema arquitetural e melhora o baseline, mas sua promoção para trading exige uma
base maior por evento, calibração e avaliação de reação real do mercado.

### 12.1. Validação de ponta a ponta na coletiva do Fed

O pipeline completo foi repetido na coletiva do FOMC de 29 de julho de 2026 com
Whisper `small.en`, gate neural e v2-large simultaneamente em CUDA. A execução levou
93,5 segundos para um vídeo de 2.704 segundos e retornou:

| Gate | Bearish | Força | Direcionais | Excluídos | Cobertura |
|---|---:|---:|---:|---:|---:|
| Heurístico | 65,83% | 0,3166 | 49 | 52 | 57,71% |
| Neural v1 | 80,66% | 0,6132 | 13 | 88 | 15,49% |

Ambos produziram `bearish` acionável, mas o modelo treinado foi substancialmente
mais seletivo: marcou 73 turnos da autoridade como neutros, um como irrelevante e
removeu 14 turnos identificados como jornalistas. O resultado completo está em
`models/policy-relevance-gate-v1/fed_video_validation.json`.

A menor cobertura é coerente com um gate conservador, porém ainda precisa ser
avaliada contra anotações humanas por turno. A validação confirma execução e
integração na GPU; ela não prova, isoladamente, que cada exclusão está correta.
