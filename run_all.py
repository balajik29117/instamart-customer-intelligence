"""Run the four project parts in dependency order."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for command in [[sys.executable,'generate_data.py'],[sys.executable,'-m','src.part_a'],[sys.executable,'-m','src.validate'],[sys.executable,'-m','src.part_b'],[sys.executable,'-m','src.part_c'],[sys.executable,'-m','src.part_d']]:
    subprocess.run(command,cwd=ROOT,check=True)
print('PROJECT PIPELINE COMPLETED')
