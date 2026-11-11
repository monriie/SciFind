"""Copy only browser assets to dist and create a deployable ZIP."""
from pathlib import Path
import shutil
import zipfile

root=Path(__file__).resolve().parent
dist=root/"dist"
dist.mkdir(exist_ok=True)
files=["index.html","app.js","bootstrap.js","retrieval.js","search-engine.js","styles.css","favicon.svg","verify.html","verify.js","NOTICE.md"]
folders=["vendor","models","data"]
for name in files:
    shutil.copy2(root/name,dist/name)
for name in folders:
    shutil.copytree(root/name,dist/name,dirs_exist_ok=True)
(dist/".nojekyll").write_text("",encoding="utf-8")
with zipfile.ZipFile(root/"scifind-static.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(dist.rglob("*")):
        if path.is_file():archive.write(path,str(path.relative_to(dist)))
size=sum(p.stat().st_size for p in dist.rglob("*") if p.is_file())
print(f"Static output: {dist}\nSize: {size/1024**2:.1f} MiB\nZIP: {root/'scifind-static.zip'}")
