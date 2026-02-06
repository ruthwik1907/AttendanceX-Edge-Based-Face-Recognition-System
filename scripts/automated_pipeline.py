# scripts/automated_pipeline.py
# -------------------------------------------
# Shorter pipeline launcher — uses sys.executable and robust logging
# Legacy export/evaluate steps removed (redundant)
# -------------------------------------------

import argparse
import subprocess
import sys
from pathlib import Path
import datetime
import shlex

parser = argparse.ArgumentParser(description="Run AttendanceX automated pipeline (uses venv python)")
parser.add_argument('--auto', action='store_true', help="Run all steps automatically")
parser.add_argument('--export-tflite', action='store_true', help="Forward flag to training (if you want tensorflow export)")
parser.add_argument('--logdir', default='logs', help="Directory to write pipeline logs")
args = parser.parse_args()

PY = sys.executable
ROOT = Path('.').resolve()
SCRIPTS = ROOT / 'scripts'
LOGDIR = ROOT / args.logdir
LOGDIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = LOGDIR / f'pipeline_{timestamp}.log'

steps = [
    ('Prepare folders from Excel', [PY, str(SCRIPTS / 'prepare_folders.py')]),
    ('Process (detect/align/augment)', [PY, str(SCRIPTS / 'process_all.py')]),
    ('Train + Evaluate (ArcFace)', [PY, str(SCRIPTS / 'train_and_evaluate.py'), '--normalize', '--test-size', '0.25']),
    ('Compute centroids & thresholds', [PY, str(SCRIPTS / 'compute_centroids_thresholds.py')]),
    ('Verify uploads', [PY, str(SCRIPTS / 'verify_uploads.py')])
]

if args.export_tflite:
    for i, (name, cmd) in enumerate(steps):
        if 'train_and_evaluate.py' in ' '.join(cmd):
            steps[i] = (name, cmd + ['--calibrate'])  # forward maybe other flags, change as needed

def run_step(name, cmd):
    cmd_display = ' '.join(shlex.quote(p) for p in cmd)
    header = f"\n==> Step: {name}\nCMD: {cmd_display}\n"
    print(header)
    with open(log_path, 'a', encoding='utf-8') as lf:
        lf.write(header)
        lf.write("\n--- stdout/stderr ---\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end='')
                lf.write(line)
            proc.wait()
            lf.write(f"\nEXIT CODE: {proc.returncode}\n\n")
        except KeyboardInterrupt:
            proc.kill()
            lf.write("\nPipeline interrupted by user.\n")
            raise
    return proc.returncode

if args.auto:
    for name, cmd in steps:
        rc = run_step(name, cmd)
        if rc != 0:
            print(f"\n Step failed: {name} (exit {rc}). See log: {log_path}\n")
            sys.exit(rc)
    print(f"\n Pipeline complete — all steps succeeded! See log: {log_path}\n")
else:
    print("Run with --auto to execute the pipeline.")
    print("This launcher writes logs to:", str(LOGDIR))
