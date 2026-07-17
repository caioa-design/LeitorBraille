"""Sistema de treinamento da rede neural"""

import random
from .rede import RedeNeural
from .braille import obter_dados_treinamento, indice_para_letra

class Treinador:
    """Responsável pelo treinamento da rede neural"""
    
    def __init__(self, rede, taxa_aprendizado=0.5):
        """
        Inicializa o treinador.
        
        Args:
            rede: Instância de RedeNeural
            taxa_aprendizado: Taxa de aprendizado (learning rate)
        """
        self.rede = rede
        self.taxa_aprendizado = taxa_aprendizado
        self.historico_erro = []
    
    def treinar_epoca(self, dados):
        """
        Treina a rede em uma época (passa por todos os dados uma vez).
        
        Args:
            dados: Lista de tuplas (padrão_braille, índice_letra)
            
        Returns:
            float: Erro médio da época
        """
        erros = []
        random.shuffle(dados)
        
        for padrao, alvo in dados:
            # Forward pass
            saida = self.rede.forward(padrao)
            
            # Calcula erro
            erro = self.rede.calcular_erro(alvo)
            erros.append(erro)
            
            # Backward pass
            self.rede.backward(alvo, self.taxa_aprendizado)
        
        erro_medio = sum(erros) / len(erros)
        self.historico_erro.append(erro_medio)
        return erro_medio
    
    def treinar(self, dados, num_epocas=1000, verbose=True):
        """
        Treina a rede por múltiplas épocas.
        
        Args:
            dados: Lista de tuplas (padrão_braille, índice_letra)
            num_epocas: Número de épocas para treinar
            verbose: Se True, imprime progresso
        """
        print(f"Iniciando treinamento por {num_epocas} épocas...\n")
        
        for epoca in range(num_epocas):
            erro = self.treinar_epoca(dados)
            
            if verbose and (epoca + 1) % 100 == 0:
                print(f"Época {epoca + 1}/{num_epocas} - Erro: {erro:.6f}")
        
        print(f"\nTreinamento concluído!")
        print(f"Erro final: {self.historico_erro[-1]:.6f}")
    
    def validar(self, dados=None):
        """
        Valida a rede em todos os dados.
        
        Args:
            dados: Dados para validação (padrão, alvo)
            
        Returns:
            float: Acurácia (0 a 1)
        """
        if dados is None:
            dados = obter_dados_treinamento()
        
        corretos = 0
        total = len(dados)
        
        for padrao, alvo in dados:
            letra, _ = self.rede.prever(padrao)
            indice_predito = ord(letra) - ord('A')
            
            if indice_predito == alvo:
                corretos += 1
        
        acuracia = corretos / total
        return acuracia
    
    def testar_letra(self, letra):
        """
        Testa o reconhecimento de uma letra específica.
        
        Args:
            letra: Letra a testar (A-Z)
        """
        from .braille import ALFABETO_BRAILLE
        
        if letra.upper() not in ALFABETO_BRAILLE:
            print(f"Letra inválida: {letra}")
            return
        
        padrao = ALFABETO_BRAILLE[letra.upper()]
        resultado, confianca = self.rede.prever(padrao)
        
        print(f"Letra: {letra.upper()}")
        print(f"Padrão: {padrao}")
        print(f"Predição: {resultado}")
        print(f"Confiança: {confianca * 100:.2f}%")
        print()
