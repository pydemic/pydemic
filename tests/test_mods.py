class TestNamespaceModules:
    """
    Prevent accidentally deleting symbols from namespace modules.
    """

    def test_pydemic_all_has_the_correct_symbols(self):
        import pydemic.all as mod

        names = set(dir(mod))
        for name in ("mundi", "mdm", "mhc", "date"):
            assert name in names

    def test_pydemic_region_has_the_correct_symbols(self):
        import pydemic.region as mod

        names = set(dir(mod))
        for name in ("hospital_capacity", "icu_capacity", "population", "age_distribution"):
            assert name in names
