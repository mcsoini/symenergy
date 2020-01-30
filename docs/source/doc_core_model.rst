.. include:: <isonum.txt>

.. _label_doc_core_model:

===========================================
The model class (symenergy.core.model)
===========================================

.. toctree::
    :name: Table of Contents
    :maxdepth: 3

    doc_core_model

.. _sect_init:

Initialize model
================

.. autoclass:: symenergy.core.model.Model


.. _sect_comps:

Add model components
================================


.. automethod:: symenergy.core.model.Model.add_slot
.. automethod:: symenergy.core.model.Model.add_slot_block
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

In order to obtain all solution of the model, it is sufficient to call the method ``symenergy.core.model.Model.generate_solve``. This reads the cached solutions from the corresponding pickle file, if applicable. For more fine-grained control over the solution steps, the individual methods listed below can be called.

.. automethod:: symenergy.core.model.Model.generate_solve
.. automethod:: symenergy.core.model.Model.init_constraint_combinations
.. automethod:: symenergy.core.model.Model.define_problems
.. automethod:: symenergy.core.model.Model.solve_all
.. automethod:: symenergy.core.model.Model.filter_invalid_solutions
.. automethod:: symenergy.core.model.Model.generate_total_costs



.. _sect_analysis:

Results analysis
================

.. automethod:: symenergy.core.model.Model.print_results
.. automethod:: symenergy.core.model.Model.get_results_dict


Other methods
================

Other important methods which are not directly called but referred to by others.

.. automethod:: symenergy.core.model.Model._get_mask_linear_dependencies



