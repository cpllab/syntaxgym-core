.. _quickstart:

Quickstart
==================

Requirements
------------

``syntaxgym`` is supported for Windows, OS X, and Linux systems. It wraps
around the `LM Zoo <https://cpllab.github.io/lm-zoo>`_ standard, which requires
`Docker <https://docs.docker.com/get-docker/>`_ to run language model images.

Installation
------------

You can install ``syntaxgym`` using the Python package manager ``pip``::

  pip install -U syntaxgym

Define your first test suite
----------------------------

Next, we'll define a simple **test suite**: an experiment which tests a
language model's knowledge of some grammatical phenomenon. The below test suite
tests knowledge of English subject--verb number agreement.

For more information on the structure of SyntaxGym test suites, see
:ref:`architecture`. The JSON standard for test suites is documented in
:ref:`suite_json`.

.. literalinclude:: sample_suite.json
  :language: JSON

Run evaluations
---------------

Now we'll evaluate the performance of a language model on our new test suite.

Pick a language model
^^^^^^^^^^^^^^^^^^^^^

SyntaxGym interfaces with `LM Zoo <https://cpllab.github.io/lm-zoo>`_ language
models. We can pick any language model, then, from the `LM Zoo registry
<https://cpllab.github.io/lm-zoo/models.html>`_. If you're interested in
evaluating your own language model in SyntaxGym, you'll first need to prepare a
model image under the LM Zoo standard. For more information, see the `LM Zoo
documentation <https://cpllab.github.io/lm-zoo/contributing.html>`_.

We'll use the model ``gpt2`` as an example here.

Command line usage
^^^^^^^^^^^^^^^^^^

You can run evaluations using the ``syntaxgym`` command-line tool. The
evaluation will return per-item prediction results given a particular language
model.

.. code-block:: bash

  $ syntaxgym run gpt2 my_suite.json
  ...
  suite                           prediction_id   item_number     result
  Sample subject--verb suite      0               1               True

The ``run`` command outputs a tab-separated list of per-item results on the
test suite.

Python API usage
^^^^^^^^^^^^^^^^

We can also trigger evaluations from Python scripts. For example, to replicate
the above evaluation:

.. code-block:: python

  from lm_zoo import get_registry
  from syntaxgym import compute_surprisals, evaluate

  # Retrieve an LM Zoo ``Model`` instance for GPT2
  model = get_registry()["gpt2"]

  # Compute region-level surprisal data for our suite.
  suite = compute_surprisals(model, "my_suite.json")

  # Check predictions given the suite containing surprisals. This returns a
  # Pandas data frame by default.
  results = evaluate(suite)

  print(results.to_csv(sep="\t"))

