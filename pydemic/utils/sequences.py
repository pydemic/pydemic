def rpartition(seq, n):
    """
    Partition sequence in groups of n starting from the end of sequence.
    """
    seq = list(seq)
    out = []
    while seq:
        new = []
        for _ in range(n):
            if not seq:
                break
            new.append(seq.pop())
        out.append(new[::-1])
    return out[::-1]


def flatten_dict(dic: dict, prefix="", sep=".") -> dict:
    """
    Flatten a nested dictionary into a flat dictionary with dotted namespace.
    """

    out = {}
    for k, v in dic.items():
        k = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, f"{k}{sep}"))
        else:
            out[k] = v
    return out


def unflatten_dict(dic: dict, sep=".") -> dict:
    """
    Invert the effect of :func:`flatten_dict`
    """

    items = list(reversed(dic.items()))
    out = {}
    while items:
        k, v = items.pop()
        *keys, last_key = k.split(sep)

        d = out
        for k in keys:
            d = d.setdefault(k, {})
        d[last_key] = v

    return out


def extract_keys(keys, dic, drop=True):
    """
    Extract keys from dictionary and return a dictionary with the extracted
    values.

    If key is not included in the dictionary, it will also be absent from the
    output.
    """

    out = {}
    for k in keys:
        try:
            if drop:
                out[k] = dic.pop(k)
            else:
                out[k] = dic[k]
        except KeyError:
            pass
    return out
