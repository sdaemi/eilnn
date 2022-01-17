#
# Definitions
#
import os
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = Path(ROOT_DIR).parent.absolute()
IMAGES_DIR = os.path.join(PARENT_DIR, "images")

print(ROOT_DIR)
print(PARENT_DIR)
print(IMAGES_DIR)