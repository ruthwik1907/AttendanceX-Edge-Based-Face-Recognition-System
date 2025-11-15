import argparse, subprocess, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--auto', action='store_true')
args = parser.parse_args()

ROOT = Path('.').resolve()
SCRIPTS = ROOT / 'scripts'

steps = [
    ('Prepare folders from Excel', ['python', str(SCRIPTS / 'prepare_folders.py')]),
    ('Combined process: detect/align/augment/report', ['python', str(SCRIPTS / 'process_all.py')]),
    ('Train embeddings + classifier', ['python', str(SCRIPTS / 'train_embeddings_classifier.py')]),
    ('Compute centroids & thresholds', ['python', str(SCRIPTS / 'compute_centroids_thresholds.py')]),
    ('Export Keras classifier & TFLite convert', ['python', str(SCRIPTS / 'export_classifier_tflite.py')]),
    ('Evaluate model', ['python', str(SCRIPTS / 'evaluate_model.py')]),
    ('Verify uploads', ['python', str(SCRIPTS / 'verify_uploads.py')])
]

if args.auto:
    for name, cmd in steps:
        print(f"\n==> Step: {name}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"Step failed: {name}")
            sys.exit(1)
    print("\nAll steps completed. Models and reports are under models/ and reports/.")
else:
    print("Run with --auto to execute the pipeline.")
