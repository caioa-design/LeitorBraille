"""Alfabeto Braille e dados de treinamento"""

ALFABETO_BRAILLE = {
    'A': [1,0,0,0,0,0],
    'B': [1,1,0,0,0,0],
    'C': [1,0,0,1,0,0],
    'D': [1,0,0,1,1,0],
    'E': [1,0,0,0,1,0],
    'F': [1,1,0,1,0,0],
    'G': [1,1,0,1,1,0],
    'H': [1,1,0,0,1,0],
    'I': [0,1,0,1,0,0],
    'J': [0,1,0,1,1,0],
    'K': [1,0,1,0,0,0],
    'L': [1,1,1,0,0,0],
    'M': [1,0,1,1,0,0],
    'N': [1,0,1,1,1,0],
    'O': [1,0,1,0,1,0],
    'P': [1,1,1,1,0,0],
    'Q': [1,1,1,1,1,0],
    'R': [1,1,1,0,1,0],
    'S': [0,1,1,1,0,0],
    'T': [0,1,1,1,1,0],
    'U': [1,0,1,0,0,1],
    'V': [1,1,1,0,0,1],
    'W': [0,1,0,1,1,1],
    'X': [1,0,1,1,0,1],
    'Y': [1,0,1,1,1,1],
    'Z': [1,0,1,0,1,1]
}

def obter_dados_treinamento():
    """
    Retorna os dados de treinamento (padrões Braille com índices).
    
    Returns:
        list: Lista de tuplas (padrão, índice_letra)
    """
    dados = []
    for i, (letra, padrao) in enumerate(ALFABETO_BRAILLE.items()):
        dados.append((padrao, i))
    return dados

def letra_para_indice(letra):
    """Converte letra para índice (A=0, B=1, ..., Z=25)"""
    return ord(letra.upper()) - ord('A')

def indice_para_letra(indice):
    """Converte índice para letra (0=A, 1=B, ..., 25=Z)"""
    return chr(ord('A') + indice)
