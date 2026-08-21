"""Inject a tiny <inertial> into the empty `proxy_base` root link of every
`*_thin_plate_proxy.urdf` so Isaac Lab imports them with a real rigid body
(same fix empty_proxy.urdf needed). Idempotent: only rewrites the empty
self-closing `<link name="proxy_base"/>` form.

Usage: python3 patch_proxies.py [DATA_ROOT]   # default /root/far_scenebot_training/data
"""
import glob
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/far_scenebot_training/data"

REPL = (
    '<link name="proxy_base">\n'
    "    <inertial>\n"
    '      <mass value="0.001"/>\n'
    '      <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>\n'
    "    </inertial>\n"
    "  </link>"
)
pat = re.compile(r'<link\s+name="proxy_base"\s*/>')

files = glob.glob(f"{ROOT}/object_folder/*_thin_plate_proxy.urdf") + glob.glob(
    f"{ROOT}/object_terrain_folder/**/*_thin_plate_proxy.urdf", recursive=True
)
patched = 0
for f in files:
    s = open(f).read()
    if pat.search(s):
        open(f, "w").write(pat.sub(REPL, s))
        patched += 1
print(f"scanned {len(files)} proxy urdfs, patched {patched}")
