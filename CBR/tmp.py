import os
import subprocess

output = subprocess.check_output(['ls', '-l']).decode()
print(output)
