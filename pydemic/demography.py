import mundi


def region(name):
    """
    Load region information from region name, ISO code, complete string, or some
    other region identifier.

    Return a Region object from the mundi package.
    """


def region_code(name):
    """
    Returns the mundi code for region. The mundi code coincides with the ISO
    code when the region is present in the ISO. Finer sub-divisions not present
    in the ISO are assigned to a unique code with a similar structure.
    """
    if hasattr(name, "code"):
        return name.code
    return region(region).code


def demography(region, kind=None):
    """
    Return the demography structure for the given region.
    """
    code = region_code()
