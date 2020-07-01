.. _suite_json:

Test suite JSON representation
==============================

This page describes SyntaxGym's standard format for representing test suites.
For more information on test suites and their basic structure, see
:ref:`architecture`.

Here's an example of a simple one-item test suite which conforms to our
standard, and which measures subject--verb number agreement knowledge:

.. literalinclude:: sample_suite.json
    :language: JSON

In the following sections we'll describe the items of this JSON standard
one-by-one. The end of this document also contains a formal JSON schema
summary.

``meta``: Test suite metadata
-----------------------------

This section contains basic archival facts such as test suite name,
description, and paper reference (if relevant).

Required fields:

``name``
    A unique identifying string name for the test suite.
``metric``
    Surprisal statistic over which this test suite specifies predictions.
    Currently the only supported metric is ``"sum"`` -- this means that, when
    multiple tokens are in the same region, a model's surprisals for each
    individual token will be summed together to form the region-level surprisal
    statistic.

Optional fields:

``reference``
    A paper reference for the content of the suite.
``comment``
    Any other comments.

``region_meta``: Region declarations
------------------------------------

This object declares the mapping from region numbers to region names. Region
numbers should form a contiguous integer range beginning at 1. These names are
only for visualization purposes; they will not be used for reference within the
test suite. We will reference regions in other parts of the test suite
specification using the region numbers.


``predictions``: Prediction declarations
----------------------------------------

Predictions state expected relations about surprisal statistics between regions
and conditions within each item. We represent predictions as arithmetic formula
strings.

Prediction string format
^^^^^^^^^^^^^^^^^^^^^^^^

The atomic units of the prediction string are **region references**,
identifying a particular region instance by its region number and condition
name using the following format::

  (<region_number>;%<condition_name>%)

For example, the prediction in this document's example test suite references
region 2 of the ``mismatch`` condition using the string ``(2;%mismatch%)``. The
surrounding parentheses are necessary for these expressions to be correctly
parsed.

The user can then use these region-condition variables in symbolic
arithmetic relationships to compare the associated surprisal values.
Our example offers the following prediction string::

  (2;%mismatch%) > (2;%match%)

stating that the total surprisal in region 2 of the ``mismatch`` condition
should be greater than the total surprisal in region 2, condition ``match``.

The following operators are available:

- Comparison operators: ``<`` and ``>`` specify hard inequality constraints.
  ``=`` specifies an approximate equality constraint.
- Arithmetic operators: ``+`` and ``-`` specify float addition and subtraction,
  respectively. For example::

    (2;%mismatch%) - (2;%match%) > 0
- Logical operators: ``&`` and ``|`` specify logical conjunction and
  disjunction, respectively. This can be used to coordinate multiple
  equalities, for example::

    (2;%mismatch%) > (2;%match%) & (2;%match%) < (2;%mismatch%)

``items``
---------

The bulk of the test suite specification consists of the actual experimental
items. These are represented as lists of region instances, nested within lists
of conditions, nested within a list of items.

Concretely, each item in the ``items`` array contains two properties:

``item_number``
    A unique identifying integer item number.
``conditions``
    A list of condition objects, specified below.

Conditions
^^^^^^^^^^

Each condition in the ``conditions`` array contains two properties:

``condition_name``
    A reference to one of the test suite's conditions. Each item object should
    have the same set of condition names.
``regions``
    A list of region contents for this particular item and condition.

Regions
^^^^^^^

Each region in the ``regions`` array consists of two properties:

``region_number``
    An integer referencing one of the test suite's regions.
``content``
    A string containing the region-level text content. This should be formatted
    as natural language (not pre-tokenized). Some regions may have no content.
    There should not be leading or trailing spaces in a region's content.

Examples
--------

The format of test suites is perhaps best learned by example.
You can find plenty of example test suites in the codebase for a recent ACL
paper using SyntaxGym: `see on GitHub
<https://github.com/cpllab/syntactic-generalization/tree/master/test_suites/json>`_.


.. jsonschema:: schemas/test_suite.json
