# 📖 Capítulos do Projeto

## Capítulo 1 - Matemática da Rede Neural

### Conceitos Fundamentais

1. **Entrada**: Vetor de 6 posições (padrão Braille)
   ```
   A = [1,0,0,0,0,0]
   B = [1,1,0,0,0,0]
   C = [1,0,0,1,0,0]
   ```

2. **Pesos**: Cada entrada tem um peso
   ```
   pesos = [0.82, -0.31, 0.75, 0.19, -0.55, 0.91]
   ```

3. **Soma Ponderada**
   ```
   soma = e₁×w₁ + e₂×w₂ + ... + e₆×w₆ + bias
   ```

4. **Função de Ativação - Sigmoid**
   ```
   σ(x) = 1 / (1 + e^-x)
   ```

5. **Erro (Cross-Entropy Loss)**
   ```
   erro = -log(saída_correta)
   ```

6. **Backpropagation**
   ```
   delta = (saída - alvo)
   novo_peso = peso + taxa_aprendizado × delta × entrada
   ```

## Capítulo 2 - Implementação da Rede

A rede está implementada em:
- `src/neuronio.py` - Neurônios individuais
- `src/camada.py` - Camadas de neurônios
- `src/rede.py` - Rede completa

### Arquitetura

```
6 entradas (Braille)
    ↓
12 neurônios (ReLU)
    ↓
26 neurônios (Softmax)
    ↓
Probabilidades de A-Z
```

## Capítulo 3 - Treinamento

Implementado em `src/treinamento.py`

### Processo

1. Forward pass: Propaga entrada
2. Calcula erro
3. Backward pass: Distribui erro
4. Atualiza pesos
5. Repete

### Como treinar

```bash
python exemplos/treinar.py
```

## Capítulo 4 - Salvamento de Pesos

Os pesos são salvos em `pesos.json` no formato:

```json
{
  "camada_0": {
    "neuronio_0": {
      "pesos": [0.82, -0.31, ...],
      "bias": 0.15
    },
    ...
  },
  "camada_1": { ... }
}
```

## Capítulo 5 - Reconhecimento de Imagens

A ser implementado em `src/imagem.py` com OpenCV

## Capítulo 6 - Leitura de Páginas

A ser implementado

## Capítulo 7 - Conversão para Voz

A ser implementado em `src/voz.py`
