#!/usr/bin/env python3
"""Script para testar a rede neural treinada"""

import sys
import os

# Adiciona o diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rede import RedeNeural
from src.braille import ALFABETO_BRAILLE

def main():
    print("="*60)
    print("LEITOR BRAILLE - TESTE DE RECONHECIMENTO")
    print("="*60)
    print()
    
    # Cria e carrega a rede
    rede = RedeNeural(arquitetura=[6, 12, 26])
    
    try:
        rede.carregar_pesos()
        print()
    except FileNotFoundError:
        print("❌ Erro: arquivo 'pesos.json' não encontrado.")
        print("Execute 'python exemplos/treinar.py' primeiro.")
        return
    
    print("="*60)
    print("TESTANDO TODAS AS LETRAS")
    print("="*60)
    print()
    
    corretos = 0
    total = 0
    
    for letra in sorted(ALFABETO_BRAILLE.keys()):
        padrao = ALFABETO_BRAILLE[letra]
        resultado, confianca = rede.prever(padrao)
        
        correto = '✅' if resultado == letra else '❌'
        
        print(f"{correto} {letra} → {resultado} ({confianca*100:.1f}%)")
        
        if resultado == letra:
            corretos += 1
        total += 1
    
    print()
    print("="*60)
    print(f"RESULTADO: {corretos}/{total} ({corretos/total*100:.1f}%)")
    print("="*60)

if __name__ == '__main__':
    main()
