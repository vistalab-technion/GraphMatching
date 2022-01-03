import pytest


class BaseTestOptimization:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        self.atol = 1e-3
        self.rtol = 1e-2
        self.n = 5
