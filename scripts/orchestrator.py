#!/usr/bin/env python3
import subprocess, os
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
guard = Path.home() / ".cache/quant-screener/last_run"
guard.parent.mkdir(parents=True, exist_ok=True)
if guard.exists() and guard.read_text().strip() == date:
    print(f"already ran today ({date})"); raise SystemExit
r = subprocess.run(["python", "scripts/screener.py"], capture_output=True, text=True)
print(r.stdout, r.stderr)
if r.returncode != 0: raise SystemExit("screener failed")
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True)
git("add", ".")
if git("diff", "--cached", "--quiet").returncode != 0:
    git("commit", "-m", f"Daily quant screen — {date}")
    git("push", "origin", "main")
    print(f"committed: {date}")
guard.write_text(date)
