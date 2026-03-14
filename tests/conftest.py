# tests/conftest.py
import sys, os
import pytest

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import api.main as main_module
from api.predict import DemandPredictor

# Inject demo predictor BEFORE TestClient is created at module level in test_api.py
main_module.predictor = DemandPredictor.demo()