=============================
SymEnergy 
=============================

Symbolic modelling of energy systems, based on SymPy.

SymEnergy provides a framework for the structured solution of cost-optimization of energy systems using 
the standard Lagrange multiplier approach. The result consists in the close-form
analytical solutions to the optimization problem. For example, the energy production
from a power plant is expressed as
a function of the vector of symbolic parameters. These solutions are evaluated
for certain parameters in order to identify relevant constraint combinations.

============
Installation
============

``pip install symenergy``

.. _label_example_minimal:

`<https://pypi.org/project/SymEnergy/>`_

=============
Documentation
=============

`<https://symenergy.readthedocs.io/>`_

============
Publication
============

  * \M. C. Soini *et al.*, Does bulk electricity storage assist wind and solar in replacing dispatchable power production?, Energy Economics, `<https://doi.org/10.1016/j.eneco.2019.104495>`_. 
  * \M. C. Soini *et al.*, On the displacement of incumbent electricity storage by more efficient storage in a stylized market model,  Journal of Energy Storage, `<https://doi.org/ 10.1016/j.est.2022.105322>`_. 
  * \M. C. Soini, Systemic Interactions of Storage in the Power System, PhD Thesis, `<https://doi.org/10.13097/archive-ouverte/unige:146422>`_. 

.. _label_example_minimal:

===============
Minimal example
===============

Investigate the dispatch of two power plants 
during a single time slot in dependence on decreasing residual load 
(increasing variable renewable energy production).

Initialize a SymEnergy model instance:

::

    from symenergy.core import model
    m = model.Model()

Add a time slot ``'t0'`` with 3000 load and 1 VRE production. VRE production 
is varied later; any finite value of this parameter is acceptable at this point to initialize the parameter.

::

    m.add_slot('t0', load=3000, vre=1)

Add two power plants: Cheap baseload power ``'n'`` with limited capacity, expensive peaker plants ``'g'`` with unconstrained power production:

::

    m.add_plant('n', vc0=1, capacity=1000)
    m.add_plant('g', vc0=2)

All Lagrange problems corresponding to this model are generated and solved.

::

    m.generate_solve()


All results of this process are stored in the model attribute ``df_comb``, indexed by the *constraint combinations* (see below). The table below shows a single line of this table. It corresponds to the case where 

* the baseload plant is not producing power output (its positivity constraint is binding ``True``). 
* the baseload plant power production is not capacity-constrained (capacity constraint not binding ``False``) 
* the peaker plant output is not zero (non-binding positivity constraint ``False``)


::
    
    m.df_comb.iloc[[0]].T 

    ===================  ==================================================================================================================================================
    act_lb_n_pos_p_t0    True
    act_lb_n_p_cap_C_t0  False
    act_lb_g_pos_p_t0    False
    lagrange             g_p_t0*vc0_g_none*w_none + lb_n_pos_p_t0*n_p_t0 + n_p_t0*vc0_n_none*w_none + pi_supply_t0*w_none*(-g_p_t0 + l_t0 - n_p_t0 - vre_scale_none*vre_t0)
    variabs_multips      [g_p_t0, lb_n_pos_p_t0, n_p_t0, pi_supply_t0]
    result               (-(-l_t0*w_none + vre_scale_none*vre_t0*w_none)/w_none, -(-vc0_g_none*w_none**2 + vc0_n_none*w_none**2)/w_none, 0, vc0_g_none)
    tc                   -vc0_g_none*(-l_t0*w_none + vre_scale_none*vre_t0*w_none)
    ===================  ==================================================================================================================================================

The ``lagrange`` value in the table above  corresponds to the Lagrange function constructed for this combination of active and inactive constraints. It is equal to the total cost (``g_p_t0*vc0_g_none*w_none + n_p_t0*vc0_n_none*w_none``), the equality constraint (energy balance/supply constraint ``-g_p_t0 + l_t0 - n_p_t0 - vre_scale_none*vre_t0`` with multiplier (shadow price) ``pi_supply_t0``), plus the only  binding constraint with the corresponding multiplier ``lb_n_pos_p_t0*n_p_t0``.


The ``variabs_multips`` are a list of all symbols/variables for which the problem was solved. In this case, these are the peaker plant power output ``g_p_t0``, the shadow price of the baseload power posivity constraint (``lb_n_pos_p_t0``), the baseload power production ``n_p_t0``, and the shadow price of the supply constraint (the "electricity price") ``pi_supply_t0``. The result column contains a list of the same variables and multipliers. Not surprisingly, the solution (not fully simplified) of the peaker power production is equal to the residual load ``l_t0 - vre_scale_none*vre_t0``; the electricity price is equal to the variable cost of peaker power production ``vc0_g_none``. The column ``tc`` is the total cost, generated by substituting the variable solutions into the total cost expression.

The SymEnergy evaluator calculates numerical results from the closed-form solutions for certain parameter values. This allows the identify the relevant constraint combinations. In the example below the model is evaluated for increasing VRE production: The value of the parameter ``vre`` of the time slot ``'t0'`` is varied in 31 steps between 0 and 3000 (residual load 0). 

::
    
    from symenergy.evaluator import evaluator
    import numpy as np

    x_vals_vre = {m.slots['t0'].vre: np.linspace(0, 3000, 31)}
    ev = evaluator.Evaluator(m, x_vals=x_vals_vre)
    ev.get_evaluated_lambdas_parallel()
    ev.expand_to_x_vals_parallel()

The final results are stored in the attribute ``ev.df_exp``. The dataframe ``ev.df_bal`` is a convenient way of accessing the optimal solution's energy balance:

::

    (ev.df_bal.query('slot not in ["global"]')
       .pivot_table(index='vre_t0', columns='func',
                       values='lambd')[['n_p_t0', 'g_p_t0', 'vre_t0', 'l_t0']].plot.area())


.. image:: minimal_balance.png
    :align: center
    :alt: minimal balance


The optimal solution thus corresponds to the case where the cheap baseload power plant produces power at maximum output, while the peaker plants cover the remaining residual load. Once the VRE production is large enough to fully replace the peaker plants, the production from baseload plants is reduced.

