# scripts/baseline_runner.py  (final)
import subprocess, sys, json
from pathlib import Path

ROOT = Path('.').resolve()
PY = sys.executable
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

configs = [
    {
        'name': 'arcface_onnx',
        'cmd': [PY, 'scripts/train_and_evaluate.py',
                '--embedder', 'arcface_onnx',
                '--models-dir', 'models',
                '--data-dir', 'data/processed',
                '--normalize',
                '--test-size', '0.25']
    },
    {
        'name': 'facenet_pytorch',
        'cmd': [PY, 'scripts/train_and_evaluate.py',
                '--embedder', 'facenet_pytorch',
                '--data-dir', 'data/processed',
                '--test-size', '0.25']
    },
    {
        'name': 'mobilefacenet_onnx',
        'cmd': [PY, 'scripts/train_and_evaluate.py',
                '--embedder', 'mobilefacenet_onnx',
                '--models-dir', 'models',
                '--data-dir', 'data/processed',
                '--test-size', '0.25']
    },
    {
        'name': 'mobilefacenet_tflite',
        'cmd': [PY, 'scripts/train_and_evaluate.py',
                '--embedder', 'mobilefacenet_tflite',
                '--models-dir', 'models',
                '--data-dir', 'data/processed',
                '--test-size', '0.25']
    }
]

results = []

for c in configs:
    print(f"\n===> Running baseline: {c['name']}")
    try:
        r = subprocess.run(c['cmd'], capture_output=True, text=True, check=False)
        out = r.stdout + "\n" + r.stderr
        print(out[:4000])

        summary_files = list((ROOT/'reports').glob('*summary*.json')) + list((ROOT/'reports').glob('summary_*.json'))
        latest = max(summary_files, key=lambda p: p.stat().st_mtime) if summary_files else None

        if latest:
            try:
                s = json.load(open(latest))
                results.append({'name': c['name'], 'summary': s, 'report_file': str(latest)})
            except Exception:
                results.append({'name': c['name'], 'stdout': out[:2000]})
        else:
            results.append({'name': c['name'], 'stdout': out[:2000], 'rc': r.returncode})

    except Exception as e:
        results.append({'name': c['name'], 'error': str(e)})

outp = REPORTS / 'baselines_summary.json'
json.dump(results, open(outp, 'w'), indent=2)
print("\nAggregated baseline results written to", outp)
