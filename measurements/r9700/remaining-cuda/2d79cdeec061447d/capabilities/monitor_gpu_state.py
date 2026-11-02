#!/usr/bin/env python3
"""Read-only sysfs trace correlated with one qualifier's benchmark children."""
import argparse
import json
from pathlib import Path
import re
import time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent',type=int,required=True)
p.add_argument('--seconds',type=float,default=300)
p.add_argument('--device',type=Path,default=Path('/sys/class/drm/card1/device'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
parent=Path(f'/proc/{a.parent}')
started=(parent/'stat').read_text().split()[21]
paths={name:str(a.device/name) for name in ['gpu_busy_percent','mem_busy_percent','pp_dpm_sclk','pp_dpm_mclk']}
hwmon=next((a.device/'hwmon').glob('hwmon*'))
paths.update({name:str(hwmon/name) for name in ['freq1_input','freq2_input','power1_average','temp1_input','temp2_input','temp3_input'] if (hwmon/name).exists()})
a.output.parent.mkdir(parents=True,exist_ok=True)
end=time.monotonic()+a.seconds
samples=0
with a.output.open('w') as out:
 out.write(json.dumps(dict(type='metadata',parent=a.parent,parent_starttime=started,paths=paths,interval_seconds=.5,read_only=True))+'\n')
 while time.monotonic()<end:
  try:
   if (parent/'stat').read_text().split()[21]!=started:break
   children=(parent/'task'/str(a.parent)/'children').read_text().split()
  except FileNotFoundError:break
  record=dict(type='sample',time_ns=time.time_ns(),children=[])
  for pid in children:
   child=Path('/proc')/pid
   try:
    command=(child/'cmdline').read_bytes().decode().split('\0')[:-1]
    log=(child/'fd/1').resolve()
    with log.open('rb') as stream:
     stream.seek(0,2)
     size=stream.tell()
     stream.seek(max(0,size-8192))
     tail=stream.read().decode(errors='replace')
    repeats=re.findall(r'^repeat=(\d+)',tail,re.M)
    record['children'].append(dict(pid=int(pid),argv=command,log=str(log),last_repeat=int(repeats[-1]) if repeats else None))
   except (FileNotFoundError,PermissionError,OSError):pass
  record['metrics']={}
  for name,path in paths.items():
   try:record['metrics'][name]=Path(path).read_text().strip()
   except OSError:pass
  out.write(json.dumps(record)+'\n')
  out.flush()
  samples+=1
  time.sleep(.5)
print('Read-only GPU trace samples:',samples,flush=True)
