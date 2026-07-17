"""Rede Neural Multicamada - Classe principal"""

import json
import math
from .camada import Camada

class RedeNeural:
    """Rede neural com múltiplas camadas"""
    
    def __init__(self, arquitetura=[6, 12, 26]):
        """
        Inicializa a rede neural.
        
        Args:
            arquitetura: Lista com número de neurônios por camada
                         [entradas, camada_oculta, saídas]
        """
        self.arquitetura = arquitetura
        self.camadas = []
        
        # Cria as camadas
        for i in range(len(arquitetura) - 1):
            num_entradas = arquitetura[i]
            num_neuronios = arquitetura[i + 1]
            
            # Última camada usa softmax
            ativacao = 'softmax' if i == len(arquitetura) - 2 else 'relu'
            
            camada = Camada(num_entradas, num_neuronios, ativacao)
            self.camadas.append(camada)
    
    def forward(self, entrada):
        """
        Propaga a entrada através da rede.
        
        Args:
            entrada: Lista com 6 valores (padrão Braille)
            
        Returns:
            list: Saídas de probabilidade para cada letra (A-Z)
        """
        saida = entrada
        for camada in self.camadas:
            saida = camada.forward(saida)
        return saida
    
    def backward(self, alvo, taxa_aprendizado):
        """
        Propaga o erro para trás (backpropagation).
        
        Args:
            alvo: Índice da letra correta (0-25)
            taxa_aprendizado: Taxa de aprendizado
        """
        # Calcula o erro na saída
        saida_rede = self.camadas[-1].saidas
        deltas = []
        
        for i in range(len(saida_rede)):
            if i == alvo:
                # Cross-entropy loss derivada
                deltas.append(saida_rede[i] - 1)  # (y - 1) para saída correta
            else:
                deltas.append(saida_rede[i])  # (y - 0) para outras
        
        # Propaga para trás
        for i in range(len(self.camadas) - 1, 0, -1):
            deltas = self.camadas[i].backward(deltas, taxa_aprendizado)
        
        # Atualiza primeira camada
        self.camadas[0].backward(deltas, taxa_aprendizado)
    
    def prever(self, entrada):
        """
        Realiza uma predição.
        
        Args:
            entrada: Lista com 6 valores (padrão Braille)
            
        Returns:
            tuple: (letra, confiança) - Ex: ('A', 0.95)
        """
        saidas = self.forward(entrada)
        indice_maximo = saidas.index(max(saidas))
        confianca = saidas[indice_maximo]
        letra = chr(ord('A') + indice_maximo)
        return letra, confianca
    
    def calcular_erro(self, alvo):
        """
        Calcula o erro de cross-entropy.
        
        Args:
            alvo: Índice da letra correta
            
        Returns:
            float: Valor do erro
        """
        saida = self.camadas[-1].saidas[alvo]
        # Evita log(0)
        saida = max(1e-7, min(1 - 1e-7, saida))
        return -math.log(saida)
    
    def salvar_pesos(self, caminho='pesos.json'):
        """
        Salva os pesos em um arquivo JSON.
        
        Args:
            caminho: Caminho do arquivo para salvar
        """
        pesos_dict = {}
        
        for i, camada in enumerate(self.camadas):
            camada_dict = {}
            for j, neuronio in enumerate(camada.neuronios):
                camada_dict[f'neuronio_{j}'] = {
                    'pesos': neuronio.pesos,
                    'bias': neuronio.bias
                }
            pesos_dict[f'camada_{i}'] = camada_dict
        
        with open(caminho, 'w') as f:
            json.dump(pesos_dict, f, indent=2)
        
        print(f"Pesos salvos em {caminho}")
    
    def carregar_pesos(self, caminho='pesos.json'):
        """
        Carrega os pesos de um arquivo JSON.
        
        Args:
            caminho: Caminho do arquivo para carregar
        """
        with open(caminho, 'r') as f:
            pesos_dict = json.load(f)
        
        for i, camada in enumerate(self.camadas):
            camada_dict = pesos_dict[f'camada_{i}']
            for j, neuronio in enumerate(camada.neuronios):
                neuronio.pesos = camada_dict[f'neuronio_{j}']['pesos']
                neuronio.bias = camada_dict[f'neuronio_{j}']['bias']
        
        print(f"Pesos carregados de {caminho}")
