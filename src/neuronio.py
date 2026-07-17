"""Neurônio - Unidade básica da rede neural"""

import math
import random

class Neuronio:
    """Representa um neurônio artificial com pesos e bias"""
    
    def __init__(self, num_entradas):
        """
        Inicializa um neurônio.
        
        Args:
            num_entradas: Número de entradas (incluindo bias)
        """
        self.num_entradas = num_entradas
        # Inicializa pesos aleatoriamente (Xavier initialization)
        limite = math.sqrt(6.0 / (num_entradas + 1))
        self.pesos = [random.uniform(-limite, limite) for _ in range(num_entradas)]
        self.bias = random.uniform(-limite, limite)
        self.saida = 0
        self.entrada = 0
        self.delta = 0  # Para backpropagation
    
    def ativar(self, entradas):
        """
        Calcula a saída do neurônio usando sigmoid.
        
        Args:
            entradas: Lista de valores de entrada
            
        Returns:
            float: Saída ativada (0 a 1)
        """
        self.entrada = sum(e * p for e, p in zip(entradas, self.pesos)) + self.bias
        self.saida = self.sigmoid(self.entrada)
        return self.saida
    
    def relu(self, x):
        """ReLU: max(0, x)"""
        return max(0, x)
    
    def relu_derivada(self, x):
        """Derivada de ReLU"""
        return 1 if x > 0 else 0
    
    def sigmoid(self, x):
        """Função sigmoid: 1 / (1 + e^-x)"""
        try:
            return 1 / (1 + math.exp(-max(-500, min(500, x))))
        except:
            return 0 if x < 0 else 1
    
    def sigmoid_derivada(self, saida):
        """Derivada de sigmoid: σ(x) * (1 - σ(x))"""
        return saida * (1 - saida)
    
    def atualizar_pesos(self, entradas, taxa_aprendizado):
        """
        Atualiza pesos e bias usando o delta calculado.
        
        Args:
            entradas: Lista de valores de entrada
            taxa_aprendizado: Taxa de aprendizado (learning rate)
        """
        for i, entrada in enumerate(entradas):
            self.pesos[i] += taxa_aprendizado * self.delta * entrada
        self.bias += taxa_aprendizado * self.delta
