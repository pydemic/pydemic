from mundi import region


def sus_macro_id(model):
    """
    Return id of SUS macro region
    """
    if model.region.alt_parents:
        refs = model.region.alt_parents.strip(";").split(";")
        refs = [*filter(lambda x: x.startswith("BR-SUS:"), refs)]
        if refs:
            return refs[0]
    return None


def sus_macro_name(model):
    """
    Return name of SUS macro region
    """
    ref = sus_macro_id(model)
    if ref:
        return region(ref).name
    return None
