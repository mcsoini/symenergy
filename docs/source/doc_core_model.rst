.. include:: <isonum.txt>
.. sectnum::

===========================================
The model class (symenergy.core.model)
===========================================

* :ref:`sect_init`
* :ref:`sect_comps`
* :ref:`sect_simps`
* :ref:`sect_run`
* :ref:`sect_analysis`


.. _sect_init:

Initialize model
================

.. autoclass:: symenergy.core.model.Model


.. _sect_comps:

Add model components
================================

.. automethod:: symenergy.core.model.Model.add_slot_block
.. automethod:: symenergy.core.model.Model.add_slot
.. automethod:: symenergy.core.model.Model.add_plant
.. automethod:: symenergy.core.model.Model.add_storage
.. automethod:: symenergy.core.model.Model.add_curtailment

.. _sect_simps:

Model simplifications
=====================

Methods which allow for simplifications on the model level.

.. automethod:: symenergy.core.model.Model.freeze_parameters

.. _sect_run:

Define constraint combinations and solve model 
==============================================

.. automethod:: symenergy.core.model.Model.generate_solve

.. _sect_analysis:

Results analysis
================

.. automethod:: symenergy.core.model.Model.print_results

