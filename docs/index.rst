.. LM Zoo documentation master file, created by
   sphinx-quickstart on Tue Dec 10 16:49:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``syntaxgym``
=============

``syntaxgym`` is a Python package which provides easy, standardized, reproducible access to targeted syntactic evaluations of language models. It replicates the core behavior of the `SyntaxGym website <http://syntaxgym.org>`_.


Quick example
-------------

You can define targeted syntactic evaluations using our standard JSON format.
Here's a simple one-item evaluation which tests language models' knowledge of
subject--verb number agreement::

  {
    "meta": {"name": "Sample subject--verb suite", "metric": "sum"},
    "predictions": [{"type": "formula", "formula": "(2;%mismatch%) > (2;%match%)"}],
    "region_meta": {"1": "Subject NP", "2": "Verb", "3": "Continuation"},
    "items": [
      {
        "item_number": 1,
        "conditions": [
          {"condition_name": "match",
           "regions": [{"region_number": 1, "content": "The woman"},
                       {"region_number": 2, "content": "plays"},
                       {"region_number": 3, "content": "the guitar"}]},
          {"condition_name": "mismatch",
           "regions": [{"region_number": 1, "content": "The woman"},
                       {"region_number": 2, "content": "play"},
                       {"region_number": 3, "content": "the guitar"}]}
        ]
      }
    ]
  }

You can then use ``syntaxgym`` to evaluate a language model's performance on
this test. Below, we evaluate GPT-2's performance on the test suite::

  $ syntaxgym run gpt2 my_suite.json
  ...
  suite                           prediction_id   item_number     result
  Sample subject--verb suite      0               1               True

We can do the same thing using a Python API:

.. code-block:: python

  from lm_zoo import get_registry
  from syntaxgym import compute_surprisals, evaluate

  model = get_registry()["gpt2"]
  suite = compute_surprisals(model, "my_suite.json")
  results = evaluate(suite)
  print(results.to_csv(sep="\t"))


Next steps
----------

For more information on getting started, please see our :ref:`quickstart` guide.

.. toctree::
   quickstart
   architecture
   suite_json
   commands
   python_api
   support
   thanks
   :hidden:

|

LM Zoo is maintained by the
`MIT Computational Psycholinguistics Laboratory <http://cpl.mit.edu/>`_.


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`search`
