#!/usr/bin/env python3
"""Download the public OpenCV sample and extract the six-second test clip."""
import hashlib,json,pathlib,subprocess,urllib.request
root=pathlib.Path(__file__).resolve().parents[1];data=root/'data/video';data.mkdir(parents=True,exist_ok=True)
source=json.loads((root/'benchmarks/fixtures/vision/video-source.json').read_text())['real_clip']
p=data/'vtest.avi'
if not p.exists():
 with urllib.request.urlopen(source['source_url'],timeout=90) as r:p.write_bytes(r.read())
if hashlib.sha256(p.read_bytes()).hexdigest()!=source['source_sha256']:raise SystemExit('Video source changed; original hash does not match')
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(p),'-t','6','-an','-vf','scale=640:-2','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-y',str(data/'people-6s.mp4')],check=True)
print('Prepared real-video fixture. Encoder versions can change MP4 bytes; source checksum and extraction parameters are pinned.')
