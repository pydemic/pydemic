import datetime


def today() -> datetime.date:
    """
    Return the date today.
    """
    return now().date()


def now() -> datetime.datetime:
    """
    Return a datetime timestamp.
    """
    return datetime.datetime.now()
