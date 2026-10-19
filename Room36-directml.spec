# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(SPECPATH).resolve()))
from build_support.room36_spec import build_room36_spec

coll = build_room36_spec("directml", globals())
