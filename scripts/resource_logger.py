# scripts/resource_logger.py
"""
Simple resource logger for inference runs (CPU, memory).
Usage: run during your inference loop to log resource usage.
"""
import time, psutil, json
from pathlib import Path
LOG=Path('reports/resource_log.jsonl')
for i in range(5):
    data={'ts':time.time(),'cpu':psutil.cpu_percent(interval=1),'mem_mb':psutil.virtual_memory().used/1024/1024}
    with open(LOG,'a') as f:
        f.write(json.dumps(data)+'\\n')
    print(data)
print('Wrote resource logs to', LOG)
