from typing import Sequence


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

    items = list(dic.items())
    items.reverse()
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


def sliced(seq, idx):
    """
    Possibly slice object if index is not None.
    """
    if idx is None:
        return seq
    else:
        try:
            return seq[idx]
        except KeyError:
            raise IndexError(f"invalid index: {idx}")


def is_seq(obj) -> bool:
    """
    Return true if object is a non-string sequence.

    See Also:
        as_seq
    """
    return not isinstance(obj, str) and isinstance(obj, Sequence)


def as_seq(obj) -> Sequence:
    """
    Force object to be a sequence.

    Non-sequence arguments and strings are converted into singleton lists.

    See Also:
        is_seq
    """
    return obj if is_seq(obj) else (obj,)
