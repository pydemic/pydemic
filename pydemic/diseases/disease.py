class Disease:
    """
    Basic interface that exposes information about specific diseases.
    """

    def mortality_table(self):
        pass

    def CFR(self, adj=None, source=None):
        pass

    def CFR_by_demography(self):
        pass

    def CFR_by_region(self, region):
        pass
