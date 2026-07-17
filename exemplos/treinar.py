#!/usr/bin/env python3
"""Script para treinar a rede neural"""

import sys
import os

# Adiciona o diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rede import RedeNeural
from src.treinamento import Treinador
from src.braille import obter_dados_treinamento

def main():
    print("="*60)
    print("LEITOR BRAILLE - TREINAMENTO DA REDE NEURAL")
    print("="*60)
    print()
    
    # Cria a rede
    rede = RedeNeural(arquitetura=[6, 12, 26])
    
    # Cria o treinador com taxa de aprendizado reduzida
    treinador = Treinador(rede, taxa_aprendizado=0.1)
    
    # Obtém dados de treinamento
    dados = obter_dados_treinamento()
    
    print(f"Arquitetura da rede: 6 entradas → 12 ocultos → 26 saídas")
    print(f"Dados de treinamento: {len(dados)} letras do alfabeto Braille")
    print(f"Taxa de aprendizado: 0.1")
    print()
    
    # Treina
    treinador.treinar(dados, num_epocas=1000, verbose=True)
    
    print()
    print("="*60)
    print("VALIDAÇÃO")
    print("="*60)
    
    # Valida
    acuracia = treinador.validar()
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print()
    
    # Testa algumas letras
    print("="*60)
    print("TESTES DE RECONHECIMENTO")
    print("="*60)
    print()
    
    for letra in ['A', 'B', 'C', 'M', 'Z']:
        treinador.testar_letra(letra)
    
    # Salva os pesos
    print("="*60)
    print("SALVANDO PESOS")
    print("="*60)
    print()
    
    rede.salvar_pesos()
    print("\n✅ Treinamento concluído com sucesso!")

if __name__ == '__main__':
    main()
