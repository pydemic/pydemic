import sidekick as sk

from .disease import Disease

covid19: Disease = sk.deferred(sk.import_later(".covid19_disease:Covid19", package=__package__))

del sk
