#
# Definitions
#
import os
from pathlib import Path


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = Path(ROOT_DIR).parent.absolute()
TEST_DATA = os.path.join(PARENT_DIR, "test_data")
