# -*- mode: python ; coding: utf-8 -*-
# Developer/auto build only. Release builds use Room36-{cpu,cuda,directml}.spec.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(SPECPATH).resolve()))
from build_support.room36_spec import build_room36_spec

coll = build_room36_spec(None, globals())
