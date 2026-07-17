# 🎓 LeitorBraille - Rede Neural do Zero

Um projeto completo de **reconhecimento de Braille usando rede neural implementada do zero**, sem TensorFlow ou PyTorch.

## 📚 O Projeto

Este é um projeto educacional e funcional que implementa:

- ✅ **Rede Neural Multicamada** (6 entradas → 12 neurônios ocultos → 26 saídas)
- ✅ **Backpropagation** completo do zero
- ✅ **Reconhecimento de 26 letras** do alfabeto Braille
- ✅ **Salvamento e carregamento** de pesos
- ✅ **Processamento de imagens** com OpenCV
- ✅ **Conversão de texto para voz** em português
- ✅ Aproximadamente **900 linhas de código Python puro**

## 🚀 Estrutura do Projeto

```
LeitorBraille/
│
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências
├── pesos.json                   # Pesos treinados (será criado)
│
├── src/
│   ├── __init__.py
│   ├── neuronio.py             # Neurônios individuais
│   ├── camada.py               # Camadas de neurônios
│   ├── rede.py                 # Rede neural completa
│   ├── treinamento.py          # Sistema de treinamento
│   ├── braille.py              # Alfabeto Braille e dados
│   ├── imagem.py               # Processamento de imagens
│   ├── camera.py               # Captura com câmera
│   └── voz.py                  # Síntese de voz
│
├── exemplos/
│   ├── treinar.py              # Script para treinar a rede
│   ├── testar.py               # Script para testar
│   ├── demo.py                 # Demonstração completa
│   └── imagens/                # Imagens de teste (opcional)
│
└── docs/
    └── capitulos.md            # Documentação dos capítulos
```

## 📖 Capítulos

1. **Capítulo 1** - Matemática da Rede Neural
2. **Capítulo 2** - Implementação da Rede (rede.py)
3. **Capítulo 3** - Treinamento (treinamento.py)
4. **Capítulo 4** - Salvamento dos Pesos
5. **Capítulo 5** - Reconhecimento de Imagens
6. **Capítulo 6** - Leitura de Páginas
7. **Capítulo 7** - Conversão para Voz

## ⚙️ Instalação

```bash
# Clone o repositório
git clone https://github.com/caioa-design/LeitorBraille.git
cd LeitorBraille

# Instale as dependências
pip install -r requirements.txt
```

## 🎯 Como Usar

### Treinar a Rede

```bash
python exemplos/treinar.py
```

### Testar Reconhecimento

```bash
python exemplos/testar.py
```

### Demo Completa

```bash
python exemplos/demo.py
```

## 💡 Como Funciona

### Fluxo de Dados

```
Imagem
   ↓
OpenCV (detecta células)
   ↓
Extrai padrão Braille [1,0,0,1,0,0]
   ↓
Rede Neural (6 → 12 → 26)
   ↓
Saída: "A" (91% de confiança)
   ↓
Acumula: "ABC..."
   ↓
Texto para Voz
   ↓
Áudio em Português
```

### Estrutura da Rede

```
Entrada (6 pontos Braille)
    ↓
Camada Oculta (12 neurônios)
    ↓
Camada Saída (26 neurônios = A-Z)
    ↓
Softmax (probabilidades)
    ↓
Melhor letra
```

## 🧮 Matemática

A rede usa:

- **Função Sigmoid**: `σ(x) = 1 / (1 + e^-x)`
- **Função ReLU (oculta)**: `f(x) = max(0, x)`
- **Softmax (saída)**: Distribuição de probabilidades
- **Cross-Entropy Loss**: Medida de erro
- **Backpropagation**: Ajuste de pesos

## 📊 Resultados Esperados

Após treinamento com ~1000 épocas:

- Acurácia: > 95%
- Reconhecimento de letras individuais: 90%+
- Palavras pequenas: 80%+

## 🔧 Tecnologias

- **Python 3.8+**
- **NumPy** (cálculos numéricos)
- **OpenCV** (processamento de imagens)
- **pyttsx3** ou **gTTS** (síntese de voz)
- **JSON** (salvamento de pesos)

## 📝 Autor

Projeto educacional criado com foco em aprendizado profundo de redes neurais.

## 📄 Licença

MIT

---

**Status**: 🚧 Em Desenvolvimento

**Próximo passo**: Criar arquivos base (neuronio.py, camada.py, rede.py)
