"""Camada - Conjunto de neurônios"""

from .neuronio import Neuronio
import math

class Camada:
    """Representa uma camada de neurônios"""
    
    def __init__(self, num_entradas, num_neuronios, ativacao='sigmoid'):
        """
        Inicializa uma camada.
        
        Args:
            num_entradas: Número de entradas para cada neurônio
            num_neuronios: Número de neurônios na camada
            ativacao: Tipo de ativação ('sigmoid', 'relu', 'softmax')
        """
        self.num_neuronios = num_neuronios
        self.num_entradas = num_entradas
        self.ativacao = ativacao
        self.neuronios = [Neuronio(num_entradas) for _ in range(num_neuronios)]
        self.saidas = []
        self.entradas = []
    
    def forward(self, entradas):
        """
        Propaga as entradas para frente na camada.
        
        Args:
            entradas: Lista de valores de entrada
            
        Returns:
            list: Saídas dos neurônios
        """
        self.entradas = entradas
        self.saidas = []
        
        if self.ativacao == 'relu':
            for neuronio in self.neuronios:
                entrada_ponderada = sum(e * p for e, p in zip(entradas, neuronio.pesos)) + neuronio.bias
                neuronio.entrada = entrada_ponderada
                neuronio.saida = max(0, entrada_ponderada)  # ReLU
                self.saidas.append(neuronio.saida)
        
        elif self.ativacao == 'softmax':
            # Primeiro calcula as saídas lineares
            entradas_brutas = []
            for neuronio in self.neuronios:
                entrada_ponderada = sum(e * p for e, p in zip(entradas, neuronio.pesos)) + neuronio.bias
                neuronio.entrada = entrada_ponderada
                entradas_brutas.append(entrada_ponderada)
            
            # Aplica softmax
            max_entrada = max(entradas_brutas)
            exp_entradas = [math.exp(x - max_entrada) for x in entradas_brutas]
            soma_exp = sum(exp_entradas)
            self.saidas = [e / soma_exp for e in exp_entradas]
            
            for i, neuronio in enumerate(self.neuronios):
                neuronio.saida = self.saidas[i]
        
        else:  # sigmoid
            for neuronio in self.neuronios:
                saida = neuronio.ativar(entradas)
                self.saidas.append(saida)
        
        return self.saidas
    
    def backward(self, deltas, taxa_aprendizado):
        """
        Propaga o erro para trás na camada.
        
        Args:
            deltas: Lista de deltas da camada seguinte
            taxa_aprendizado: Taxa de aprendizado
            
        Returns:
            list: Deltas para a camada anterior
        """
        for i, neuronio in enumerate(self.neuronios):
            neuronio.delta = deltas[i]
            neuronio.atualizar_pesos(self.entradas, taxa_aprendizado)
        
        # Calcula deltas para a camada anterior
        deltas_anteriores = [0] * self.num_entradas
        for i, neuronio in enumerate(self.neuronios):
            for j in range(self.num_entradas):
                deltas_anteriores[j] += neuronio.delta * neuronio.pesos[j]
        
        return deltas_anteriores
