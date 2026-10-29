"""Build the optional public-inventory solver without changing Python packages."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compiler',default='g++')
args=parser.parse_args()
if os.name!='posix':
    raise SystemExit('The optional build currently supports Linux; the Python solver remains available.')
compiler=shutil.which(args.compiler)
if compiler is None:
    raise SystemExit(f'Compiler {args.compiler!r} not found; the Python solver remains available.')
package=Path(__file__).resolve().parents[1]/'src/junqi'
source=package/'_inventory_native.cpp'
target=package/('_inventory_native'+sysconfig.get_config_var('EXT_SUFFIX'))
temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
try:
    subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',
                    '-I'+sysconfig.get_paths()['include'],str(source),'-o',str(temporary)],check=True)
    os.replace(temporary,target)
finally:
    temporary.unlink(missing_ok=True)
print(target)
