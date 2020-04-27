=======
Pydemic
=======

The Pydemic project implements several tools that help simulating and understanding epidemic outbreaks.
The main focus right now is modelling COVID-19, and it integrates with databases with relevant
demographic and epidemiological information from many countries in the world.

This library implements a few classic epidemiological models such as SIR and SEIR and introduces
a few new models such as SEAIR and SEICHAR. (yet to be published, we will link the preprint here).

Usage
=====

You can run models from the command line::

$ python -m pydemic.models.seir

Or, more typically, from Python code

>>> from pydemic.models import SEIR
>>> m = SEIR(demography='Italy')
>>> run = m.run(120)
>>> run.plot(show=True)


Installation
============

Either clone this repository and install locally using `flit install -s` or use
`pip install pydemic`. If you do not have flit, install it using either your distribution
package manager or `pip install flit --user` before continuing.
