# Copyright 2017 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    # flake8 forks worker processes. Forking this pytest process after the ROS integration
    # tests (DDS + executor threads alive) can deadlock the children, so lint in a fresh
    # interpreter instead of calling ament_flake8's main_with_errors() in-process.
    proc = subprocess.run(
        [sys.executable, '-c', 'import sys; from ament_flake8.main import main; sys.exit(main())'],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, 'Found code style errors / warnings:\n' + proc.stdout
