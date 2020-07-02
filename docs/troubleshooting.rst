.. _troubleshooting:

Troubleshooting
===============

This page lists some common issues encountered by SyntaxGym users. For in-depth
support, please see :ref:`support`.

Commands return ``DockerError``
-------------------------------

When attempting to run SyntaxGym evaluations or retrieve surprisals, you may
encounter the following error::

  lm_zoo.errors.BackendConnectionError: Backend DockerBackend encountered error: ('Connection aborted.', ConnectionRefusedError(111, 'Connection refused'))

**Solution:** SyntaxGym requires Docker to run properly. Verify that you have
`installed Docker <https://docs.docker.com/get-docker/>`_ and that the Docker
service is running.

If you are using OS X or Windows and are not a terminal expert, we recommend
installing `Docker Desktop <https://www.docker.com/products/docker-desktop>`_.
