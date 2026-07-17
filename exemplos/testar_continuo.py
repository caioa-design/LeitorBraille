#!/usr/bin/env python3
"""Script para testar continuamente"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rede import RedeNeural
from src.braille import ALFABETO_BRAILLE

def main():
    rede = RedeNeural(arquitetura=[6, 12, 26])
    
    try:
        rede.carregar_pesos()
    except FileNotFoundError:
        print("❌ Erro: arquivo 'pesos.json' não encontrado.")
        print("Execute 'python exemplos/treinar.py' primeiro.")
        return
    
    print("="*60)
    print("TESTE CONTÍNUO - Pressione Ctrl+C para parar")
    print("="*60)
    print()
    
    contador = 0
    
    try:
        while True:
            print(f"\n🔄 Rodada {contador + 1}")
            print("-" * 60)
            
            corretos = 0
            for letra in sorted(ALFABETO_BRAILLE.keys()):
                padrao = ALFABETO_BRAILLE[letra]
                resultado, confianca = rede.prever(padrao)
                
                if resultado == letra:
                    corretos += 1
                    print(f"✅ {letra}")
                else:
                    print(f"❌ {letra} (previu {resultado})")
            
            print("-" * 60)
            print(f"Acurácia: {corretos}/26 ({corretos/26*100:.1f}%)")
            
            contador += 1
            print(f"\nPróxima rodada em 5 segundos... (Ctrl+C para parar)")
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Teste interrompido!")
        print(f"Total de rodadas: {contador}")

if __name__ == '__main__':
    main()
