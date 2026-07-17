"""LeitorBraille - Rede Neural para Reconhecimento de Braille"""

__version__ = "1.0.0"
__author__ = "caioa-design"

from .rede import RedeNeural
from .braille import ALFABETO_BRAILLE
from .treinamento import Treinador

__all__ = ['RedeNeural', 'ALFABETO_BRAILLE', 'Treinador']
