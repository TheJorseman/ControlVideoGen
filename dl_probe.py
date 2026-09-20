import time

from core import config, model_manager

s = config.load_settings()
config.apply_env(s["model_dir"])
model_manager.reset_progress()
t = model_manager.start_download(s["model_dir"], "preproc_depth", s.get("hf_token", ""))
print("START", t)
for i in range(90):
    time.sleep(2)
    p = model_manager.get_progress()
    if i % 5 == 0:
        print("tick", i, p["status"], f"{p['files_done']}/{p['files_total']}",
              round(p["bytes_done"] / 1e6, 1), "MB", f"{p['fraction'] * 100:.0f}%")
    if p["status"] in ("done", "error"):
        print("FINAL", p["status"], p.get("error"))
        break

import glob
n = len(glob.glob(r"F:\ControlVideoGen\models\preproc_depth\**\*", recursive=True))
print("files:", n)
