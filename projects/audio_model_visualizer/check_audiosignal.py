import sys
sys.path.insert(0, 'descript-audio-codec')
from audiotools import AudioSignal
import inspect

sig = inspect.signature(AudioSignal.__init__)
print('AudioSignal.__init__ Signature:')
print(sig)
print()
print('Parameters:')
for name, param in sig.parameters.items():
    if name != 'self':
        print(f'  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else "no annotation"}')
        if param.default != inspect.Parameter.empty:
            print(f'    default: {param.default}')
