from pydemic.testing import en


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark tests that take long to run")
    config.addinivalue_line(
        "markers", "external: mark tests that use external resources from the internet"
    )
