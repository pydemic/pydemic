.. image:: https://api.codeclimate.com/v1/badges/43969668ddbe90e8c4e6/maintainability
   :target: https://codeclimate.com/github/pydemic/pydemic/maintainability
   :alt: Maintainability

.. image:: https://api.codeclimate.com/v1/badges/43969668ddbe90e8c4e6/test_coverage
   :target: https://codeclimate.com/github/pydemic/pydemic/test_coverage
   :alt: Test Coverage


==============
Pydemic Models
==============

The Pydemic Models project implements several tools that help simulating and understanding epidemic outbreaks.
The main focus right now is modelling COVID-19, and it integrates with databases with relevant
demographic and epidemiological information from many countries in the world.

This library implements a few classic epidemiological models such as SIR and SEIR and introduces
a few new models such as SEAIR and SEICHAR. (yet to be published, we will link the preprint here).


Basic usage and installation
============================

Install it with `pip install pydemic-models`. If you want to contribute to the project, clone this repository
and install locally using `flit install -s`. If you do not have flit in your computer, install
it using either your distribution package manager or `pip install flit --user` before continuing.

Once pydemic is installed, you can run simple simulations from the command line::

$ python -m pydemic.models.seir

(run it with a --help flag to understand how to tweak the simulation parameters)

More typically, you would prefer to control it from Python code

>>> from pydemic.models import SEIR
>>> m = SEIR(demography='Italy')
>>> run = m.run(120)
>>> run.plot(show=True)


Getting started
===============

This package implements several different epidemiological models that are relevant to COVID-19
and other transmissible diseases. The most simple ones are in the class of exponential models
with closed form solutions, and are usually good approximations at the outset of an epidemic.

The most simple of those, the linearized SIR model, has a simple exponential solution for the
infectious compartment. Let us create a simple example, and show how it works on code.

>>> from pydemic.models import SIR
>>> m = SIR()

By default, it creates a model with a population of 1 and contaminated with a seed of 1
infectious per million. We can simulate this scenario by calling the run(dt) method with
the desired time interval.

>>> m.run(60)

The results of the simulation are exposed as pandas dataframes and can be easily
processed, plotted, and analyzed. The time series for any component of the model can be
extracted using Python's indexing notation

>>> infectious = m["infectious"]  # A Pandas data frame
>>> infectious.plot()
...

Pydemic uses a clever notation that allow us to make convenient transformations in the
resulting components by simply appending the desired transformations after the
component name.

>>> m["infectious:weeks"]  # A Pandas dataframe indexed by weeks instead of days
...

It also recognizes the shorthand notation for each compartment and allows some advanced
indexing tricks such as slicing and retrieving several columns at once.

>>> m[["S", "I", "R"]]
...


Model parameters
----------------

The model exposes all relevant epidemiological parameters as attributes.

>>> m.R0, m.infectious_period
(2.74, 3.64)

Some parameters can be naturally expressed in different equivalent mathematical forms,
or have common aliases or shorthand notations. Pydemic makes sure to keep everything
in sync.

>>> m.infectious_period = 4
>>> m.gamma
0.25

The parameters can also be accessed using the special ``.params`` attribute, which
exposes the complete list of parameters in a normalized and convenient form. The
``.params`` attribute also stores information about confidence intervals and the
reference used to assign the given point value, when available.

>>> m.params.table()
...
