import pytest
import os
import cometpy.cfg

@pytest.fixture
def data_rank2_path():
    return os.path.join(os.path.dirname(__file__), 'data', 'test_rank2.mtx')

@pytest.fixture
def data_rank2_transpose_path():
    return os.path.join(os.path.dirname(__file__), 'data', 'test_rank2_transpose.mtx')

@pytest.fixture
def data_tc_path():
    return os.path.join(os.path.dirname(__file__), 'data', 'tc.mtx')

gpu = pytest.mark.skipif(not cometpy.cfg.gpu_target_enabled, reason="GPU support not enabled")