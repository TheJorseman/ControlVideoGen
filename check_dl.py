import time
from pathlib import Path

from huggingface_hub import HfApi

from core import config, model_manager

api = HfApi()
for repo, key in [("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan_ti2v_5b")]:
    remote = [f for f in api.list_repo_files(repo) if not f.startswith(".git")]
    dest = Path(config.load_settings()["model_dir"]) / key
    missing = [f for f in remote if not (dest / f).exists()]
    print(f"5B check: remote={len(remote)} missing={len(missing)}", missing[:5])

s = config.load_settings()
config.apply_env(s["model_dir"])
t = model_manager.start_download(s["model_dir"], "preproc_depth", s.get("hf_token", ""))
print("START", t)
last = None
for i in range(90):
    time.sleep(2)
    p = model_manager.get_progress()
    line = (p["status"], f"{p['files_done']}/{p['files_total']}",
            round(p["bytes_done"] / 1e6, 1), f"{p['fraction'] * 100:.0f}%")
    if line != last:
        print("tick", line)
        last = line
    if p["status"] in ("done", "error"):
        print("FINAL", p["status"], p.get("error"))
        break
