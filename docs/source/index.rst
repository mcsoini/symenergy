.. symenergy documentation master file, created by

   sphinx-quickstart on Sun Feb 17 09:54:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================
Welcome to the SymEnergy docs
=============================

Symbolic modelling of energy systems, based on SymPy.

SymPy provides a framework for the structured solution of cost-optimization of energy systems using 
the standard Lagrange multiplier approach. The result consists in the close-form
analytical solutions to the optimization problem. For example, the energy production
from a power plant is expressed as :math:`p=p\left(\mathbf{p}\right)`, i.e. as
a function of the vector of parameters :math:`\mathbf{p}`. These solutions are evaluated
for certain parameters in order to identify relevant constraint combinations.

Basic example
=============

nimal example: Investigate the dispatch of two power plants 
during a single time slot in dependence on decreasing residual load 
(increasing variable renewable energy production).

Initialize a SymEnergy model instance:

::

    from symenergy.core import model
    m = model.Model()

Add a time slot ``'t0'`` with 3000 load and 1 VRE production. VRE production 
is varied later; any finite value of this parameter is acceptable at this point.

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

The ``lagrange`` column corresponds to the Lagrange function constructed for this combination of active and inactive constraints. It is equal to the total cost (``g_p_t0*vc0_g_none*w_none + n_p_t0*vc0_n_none*w_none``), the equality constraint (energy balance/supply constraint ``-g_p_t0 + l_t0 - n_p_t0 - vre_scale_none*vre_t0`` with multiplier (shadow price) ``pi_supply_t0``), plus the binding constraint with the corresponding multiplier ``lb_n_pos_p_t0*n_p_t0``.

===================  ==================================================================================================================================================
act_lb_n_pos_p_t0    True
act_lb_n_p_cap_C_t0  False
act_lb_g_pos_p_t0    False
lagrange             g_p_t0*vc0_g_none*w_none + lb_n_pos_p_t0*n_p_t0 + n_p_t0*vc0_n_none*w_none + pi_supply_t0*w_none*(-g_p_t0 + l_t0 - n_p_t0 - vre_scale_none*vre_t0)
variabs_multips      [g_p_t0, lb_n_pos_p_t0, n_p_t0, pi_supply_t0]
result               (-(-l_t0*w_none + vre_scale_none*vre_t0*w_none)/w_none, -(-vc0_g_none*w_none**2 + vc0_n_none*w_none**2)/w_none, 0, vc0_g_none)
tc                   -vc0_g_none*(-l_t0*w_none + vre_scale_none*vre_t0*w_none)
===================  ==================================================================================================================================================

The ``variabs_multips`` are a list of all symbols/variables for which the problem was solved. In this case, these are the peaker plant power output ``g_p_t0``, the shadow price of the baseload power posivity constraint (``lb_n_pos_p_t0``), the baseload power production ``n_p_t0``, and the shadow price of the supply constraint (the "electricity price") ``pi_supply_t0``. The result column contains a list of the same variables and multipliers. Not surprisingly, the solution (not fully simplified) of the peaker power production is equal to the residual load ``l_t0 - vre_scale_none*vre_t0``; the elelectricity price is equal to the variable cost of peaker power production ``vc0_g_none``. The column ``tc`` is the total cost.

The SymEnergy evaluator calculates numerical results from the closed-form solutions for certain parameter values. This allows the identify the relevant constraint combinations.

::
    
    from symenergy.evaluator import evaluator






Description of the solution process
===============================

The figure below 
provides an overview of the solution process: 

* The model components are defined (time slots, power plants, storage, ...)
* The optimal operation of these components (power production, charging, capacity
  retirements) is limited by certain constraints (power capacity of power plants
  and storage, energy capacity of storage, energy balance constraint for the whole system, etc.).
* Some of these constraints are equality constraints (e.g. energy balance, must hold in all cases);
  others are inequality constraints (e.g. capacity constraints: a power plant can produce power 
  at the capacity limit, but it can possibly operate at lower output).
* **Any solution to the optimization problem will hold for a specific combination of binding and 
  non-binding inequality constraints.** For example, two constraint
  combinations might be differ by the capacity constraint of a single power plant being binding or
  non-binding, all things equal. 
* Following the standard Lagrange approach, the Lagrange function can be defined for each of these constraint combinations :math:`\mathrm{CC}`

        .. math::

           \mathcal{L}_\mathrm{CC}(\mathbf{p}, \mathbf{v}, \mathbf{\pi}, \mathbf{\lambda}) =c_\mathrm{Total}(\mathbf{p},\mathbf{v}) + \sum_i \pi_i \Pi_i(\mathbf{p}, \mathbf{v}) + \sum_j \lambda_{j} \Lambda_{j,\mathrm{CC}}(\mathbf{p}, \mathbf{v})

with the inequality constraint  :math:`\Lambda_{j,\mathrm{CC}}(\mathbf{p}, \mathbf{v})`, the equality constraints :math:`\Pi_i(\mathbf{p}, \mathbf{v})`, and the total system cost  :math:`c_\mathrm{Total}(\mathbf{p},\mathbf{v})`. :math:`\mathbf{v}` and :math:`\mathbf{p}` are the vectors of all variables and parameters, respectively. 
* The total cost function :math:`c_\mathrm{Total}` is quadratic (see below). Therefore,
  the Karush-Kuhn-Tucker (KKT) conditions :math:`\nabla_{\mathbf{v},\mathbf{\pi},\mathbf{\lambda}}\mathcal{L}_\mathrm{CC} = 0`
  form a system of linear equations, whose closed-form analytical solutions :math:`\mathrm{v}(\mathbf{p})`, :math:`\mathrm{\pi}(\mathbf{p})`, 
  and :math:`\mathbf{\lambda}(\mathbf{p})` can be calculated using the corresponding SymPy function.


Components and their constraints and costs
===========

.. image:: _static/flowchart_symenergy.png
    :align: center
    :alt: flowchart

.. important::
   This documentation is work in progress.

===========================

.. toctree::
   :name: Table of Contents
   :maxdepth: 3
   :caption: Contents:

   example_constant
   symenergy_doc_cookbook
   doc_core_model
   doc_core_component
   doc_core_collections
   doc_auxiliary_io
   doc_evaluator_plotting


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
