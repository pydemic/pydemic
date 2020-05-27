import datetime


def today(n=0) -> datetime.date:
    """
    Return the date today.
    """
    today = now().date()
    if n:
        return today + datetime.timedelta(days=n)
    return today


def now() -> datetime.datetime:
    """
    Return a datetime timestamp.
    """
    return datetime.datetime.now()
