.. _suite_json:

Test suite JSON representation
==============================

Test suites lie at the heart of psycholinguistic evaluation.
The items in a test suite are given as input to a language model,
and the resulting surprisal values are used to assess the model's performance.
However, test suites from existing papers have been published in a variety of formats,
making them difficult to adapt across models and evaluation pipelines.
We require our test suites to conform to a standardized format to facilitate
replications.

.. _meta:

Meta information
-------------------

First, we require some meta information about the suite, which will
be displayed on the SyntaxGym website. In addition to basic archival facts
such as suite name, description, and reference, we also support tags.
Tags allow users to identify collections of suites that may probe similar
syntactic phenomena, such as filler-gap dependencies or number agreement.

Users may also put additional information in a ``comment`` field,
which will not be displayed on the SyntaxGym website but will be visible to
anyone who downloads the test suite as a JSON file.

.. We also require a **metric**, which specifies the way surprisal
.. should be aggregated over the tokens within a region
.. (see the :ref:`regions` section for more details).
.. Currently, the supported metrics are ``sum``, ``mean``, ``median``, ``range``,
.. ``max``, and ``min``. Users may specify any individual metric or any subset
.. of these metrics.

.. For users uploading test suites in JSON format, the metric must be specified
.. as a string or list of strings. You can specify any individual metric as a
.. string (e.g. ``'range'``), multiple metrics as a list (e.g. ``['sum', 'mean']``),
.. or all metrics as the string ``'all'``.

An example ``meta`` dict in JSON looks like this:

.. code-block:: json
    :linenos:

    {
        "meta": {
            "name": "test",
            "tags": ["pedagogical"],
            "author": "Syntax James",
            "description": "An example test suite for the SyntaxGym documentation.",
            "reference": "James, Syntax (1956). Syntactic Strucgyms. MIT Press.",
            "comment": "Syntax James is a figment of our imagination."
        }
    }

.. _regions:

Regions
-------------------

The atomic unit of a test suite is a **region**. A region is a
chunk of a sentence that we are interested in comparing across conditions.
Regions are defined separately from the items themselves, and each sentence in
a test suite is partitioned into the same regions.

For example, suppose we are designing a test suite for subject-verb number
agreement. One might imagine that a researcher would be interested in comparing
surprisal values at the **verb** of the sentence when number agreement is
satisfied or violated. Thus, one natural region would be ``verb``. Assuming
we're dealing with very simple sentences, we might then define a region
``subject`` to deal with the content before the verb and ``post-verb`` to deal
with the content after the verb, such as adverbs or final punctuation.

.. note::
    A region can consist of multiple tokens. For example, the region corresponding
    to a noun phrase might contain the tokens ``my neighbor``,
    ``the very wrinkly raisin``, or ``daffodils``.

The surprisal of a region is calculated by taking the **sum** of
the surprisals of each token in the region. In future releases, we may allow
users to specify arbitrary metrics to aggregate token-level surprisals.

.. There are certain regions where we want to compare surprisal values across
.. conditions, such as the main verb. So, we split each sentence into
.. three regions: one for the subject, one for the main verb,
.. and one for the rest of the sentence.

.. We can add a region through the web interface like so:

.. .. figure:: ../img/screenshots/web_region.png
..     :width: 90%
..     :align: center

Region JSON format
********************

Users uploading a test suite in JSON format specify this information
in a dictionary called ``region_meta`` that associates *region numbers* with
*region names*. This mapping will be used in the :ref:`predictions`
and :ref:`items`.

.. code-block:: json
    :linenos:

    {
        "region_meta": {
            "1": "subject",
            "2": "verb",
            "3": "post-verb"
        }
    }

Conditions
-------------------

In our example, suppose we are interested in two experimental conditions:
a singular subject paired with a singular verb, and
a singular subject paired with a plural verb.
Let's define these conditions as ``number_match`` and
``number_mismatch``, respectively.

Web interface users must enter the names of each condition in the test suite.
However, JSON users do not need to explicitly define condition names. They are
inferred from the :ref:`items`.

.. _predictions:

Predictions
-------------------

Researchers typically design test suites with a hypothesis in mind: if a model
has learned the correct syntactic generalization, then the surprisal at
certain regions should be greater in some conditions than others. In order to
encode these hypotheses, we allow users to specify predicted relationships
between region-level surprisal values across conditions.

Let's return to the running example. If the model has correctly learned
generalizations about number agreement in English grammar, then we would expect
the aggregate surprisal at region ``2`` (``verb``) to be higher in the
``number_mismatch`` condition than in the ``number_match`` condition.

Prediction string format
*************************

Users uploading a test suite in JSON format must include a ``predictions``
field, which contains a list of strings, each one containing a single prediction.
The format of the prediction string is very flexible and conforms to the
metagrammar specified below.

.. warning::

    Do not omit ``predictions`` in the JSON file. If you do not wish to encode
    any predictions, simply pass an empty list. The ``predictions`` field must
    be present to be properly parsed.

The atomic units of the prediction string are **variables** built with
region numbers and condition names (as specified in the :ref:`meta`).
For example, if we are interested in the surprisal at region ``2`` in the
``number_mismatch`` condition, then the variable would be formatted as
``(2;%number_match%)``.

The user can then use these region-condition variables in symbolic
mathematical relationships to compare the associated surprisal values.
For example, to encode the prediction that the surprisal at region ``2`` is
greater in the ``number_mismatch`` condition than the ``number_match`` condition,
we could use the prediction string

.. code-block:: json
    :linenos:

    {
        "predictions": ["(2;%number_mismatch%) > (2;%number_match%)"]
    }

or, equivalently,

.. code-block:: json
    :linenos:

    {
        "predictions": ["(2;%number_mismatch%) - (2;%number_match%) > 0"]
    }

and so on. The prediction string is ultimately passed to Python's ``eval``
function, so users can specify any valid Python code to manipulate the variables.

To sum up, these are the basic components of the prediction metagrammar:

- Variables are enclosed by parentheses
- Region number and condition name are separated with a semicolon
- Condition name is marked on either side by a ``%`` sign
- Order of operations is specified with square bracketing
- String is ultimately passed to Python's ``eval`` function, allowing for complex expressions

Here are some more advanced examples:

- ``[(5;%cond1%)+(3;%cond1%)] / 2] < [[(1;%cond2%)-(6;%cond2%)] / 4]`` predicts that the sum of regions ``5`` and ``3`` in ``cond1`` divided by 2 is less than the difference of regions ``1`` and ``6`` in ``cond2`` divided by 4
- ``abs((1;%cond1%) - (1;%cond2%)) > 3`` predicts that the absolute value (computed with the Python ``abs`` function) of the difference of region ``1`` in ``cond1`` vs. ``cond2`` is greater than 3

Logical operators
+++++++++++++++++++++++++

Users can also encode **logical relationships** between multiple predictions
following the syntax of `Boolean indexing in Pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing>`_.

For example, consider the following predictions:

.. code-block:: json
    :linenos:

    {
        "predictions": [
            "[(1;%cond1%) > (1;%cond2%)] & [(1;%cond3%) < (1;%cond4%)]",
            "[(2;%cond1%) > (2;%cond2%)] | [(2;%cond3%) < (2;%cond4%)]",
            "~[[(3;%cond1%) > (3;%cond2%)]"
        ]
    }

Using ``&``, ``|``, and ``~``, these three higher-order predictions encode a
**conjunction**, **disjunction**, and **negation** of predictions, respectively.

.. _items:

Items
-------------------

Finally, users must specify a list of **items**. This is the meat of
the test suite, in the sense that it provides the actual sentences that are
sent to the model for evaluation.

An item is characterized by the lexical content, and takes different forms
across conditions. For example, ``The boy swims today.`` and
``The boy swim today.`` are different instances of the same item under the
``number_match`` and ``number_mismatch`` conditions, respectively.

.. note::

    The content of each item is expected to be natural language text,
    **prior to tokenization**. Tokenization will be performed on a
    model-by-model basis when the test suite is used for evaluation.

In the web interface, users enter sentences in a grid of text boxes, where
each row corresponds to a **sentence** (a particular item under a particular
condition) and each column corresponds to a region.

Item JSON format
********************

In the JSON format, ``items`` is a list of dictionaries. Each item dictionary
specifies an ``item_number``, as well as a list of conditions.
Each dictionary in the ``conditions`` list corresponds to a sentence.
This sentence-level dictionary requires a ``condition_name`` as well as a list
of regions, where each region is represented as a dictionary with a
``region_number`` (consistent with ``region_meta``) and ``content``.
This ``content`` dictionary is where the actual text lives.

Here are example items in JSON format:

.. code-block:: json
    :linenos:

    {
        "items": [
            {
                "item_number": 1,
                "conditions": [
                    {
                        "condition_name": "number_match",
                        "regions": [
                            {
                                "region_number": 1,
                                "content": "The boy"
                            },
                            {
                                "region_number": 2,
                                "content": "swims"
                            },
                            {
                                "region_number": 3,
                                "content": "today."
                            }
                        ]
                    },
                    {
                        "condition_name": "number_mismatch",
                        "regions": [
                            {
                                "region_number": 1,
                                "content": "The boy"
                            },
                            {
                                "region_number": 2,
                                "content": "swim"
                            },
                            {
                                "region_number": 3,
                                "content": "today."
                            }
                        ]
                    }
                ]
            }
        ]
    }

Examples
-------------------

The format of test suites is perhaps best learned by example.
To view or download items from existing test suites,
see the `Test Suites page <http://alpha.syntaxgym.org/test_suite/>`_ for inspiration.ODO


.. jsonschema:: schemas/test_suite.json
