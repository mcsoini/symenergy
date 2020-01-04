
Example 1: Two plants, constant supply curves
=============================================

--------------

This illustrates storage impact in a 2 power plant/2 time slot system
with constant supply curves.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import symenergy.core.model as model
    import symenergy.evaluator.evaluator as evaluator
    
    pd.options.mode.chained_assignment = None


.. parsed-literal::

    > 11:02:07 - WARNING - symenergy.core.model - !!! Monkey-patching sympy.linsolve !!!


Initialize model
----------------

The model structure is initialized.

Parameter values are insignificant at this stage as long as they are
!=None. They represent default values and define the model structure.

``Model.generate_solve()`` loads the solved model results from the
corresponding pickle file if a model with the same structure (variables
and multipliers) has been solved before.

``nthread`` is the number of cores used for parallelized solving.

.. code:: ipython3

    from symenergy import logger
    logger.setLevel('DEBUG')
    m = model.Model(curtailment=True, nthreads=7)
    
    m.add_slot(name='day', load=4.5, vre=3)
    m.add_slot(name='night', load=5, vre=0.5)
    
    m.add_plant(name='n', vc0=10, capacity=3500, fcom=9, cap_ret=True)
    m.add_plant(name='g', vc0=90)
    
    m.add_storage(name='phs', eff=0.75, capacity=1, energy_capacity=1,
                  slots_map={'chg': ['day'] , 'dch': ['night']})
    
    #m.cache.delete()
    m.generate_solve()


.. parsed-literal::

    > 11:02:41 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:41 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:41 - INFO - symenergy.core.asset - Variable C_ret has time dependence False
    > 11:02:41 - INFO - symenergy.core.asset - Variable C_ret has time dependence False
    > 11:02:41 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:41 - INFO - symenergy.core.asset - Variable C_ret has time dependence False
    > 11:02:41 - DEBUG - symenergy.core.model - Auto-adding curtailment
    > 11:02:41 - DEBUG - symenergy.core.model - _add_curtailment with slots={'day': Slot `day`, 'night': Slot `night`}
    > 11:02:41 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:41 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:42 - DEBUG - symenergy.core.model - Auto-adding curtailment
    > 11:02:42 - DEBUG - symenergy.core.model - _add_curtailment with slots={'day': Slot `day`, 'night': Slot `night`}
    > 11:02:42 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - WARNING - symenergy.assets.storage - phs: Moving variable e from VARIABS_TIME to VARIABS
    > 11:02:42 - INFO - symenergy.core.asset - Variable pchg has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable pdch has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable e has time dependence False
    > 11:02:42 - INFO - symenergy.core.asset - Variable pchg has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable e has time dependence False
    > 11:02:42 - INFO - symenergy.core.asset - Variable pdch has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable pchg has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable pdch has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable e has time dependence False
    > 11:02:42 - DEBUG - symenergy.core.model - Auto-adding curtailment
    > 11:02:42 - DEBUG - symenergy.core.model - _add_curtailment with slots={'day': Slot `day`, 'night': Slot `night`}
    > 11:02:42 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:42 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 11:02:42 - INFO - symenergy.core.component - Generating constraint combinations for "n"
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Power plant output not simult. max end zero"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [(('act_lb_n_pos_p_day', True), ('act_lb_n_p_cap_C_day', True)), (('act_lb_n_pos_p_night', True), ('act_lb_n_p_cap_C_night', True))]
    > 11:02:42 - INFO - symenergy.core.component - Generating constraint combinations for "g"
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Power plant output not simult. max end zero"
    > 11:02:42 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('p_cap_C',)
    > 11:02:42 - INFO - symenergy.core.component - Generating constraint combinations for "day"
    > 11:02:42 - INFO - symenergy.core.component - Generating constraint combinations for "night"
    > 11:02:42 - INFO - symenergy.core.component - Generating constraint combinations for "phs"
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Full storage can`t charge"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'last', 'this'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "No simultaneous non-zero charging and non-zero discharging"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "No simultaneous full-power charging and full-power discharging"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Storage energy not simult. full and empty"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Storage charging not simult. max end zero"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 1 column combinations: [(('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pchg_cap_C_day', True))]
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Storage discharging not simult. max end zero"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 1 column combinations: [(('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pdch_cap_C_night', True))]
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All charging zero -> each discharging cannot be non-zero"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 1 column combinations: [[('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_pdch_night', False)]]
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All discharging zero -> each charging cannot be non-zero"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 1 column combinations: [[('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_pchg_day', False)]]
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All charging zero -> each energy cannot be non-zero"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All discharging zero -> each energy cannot be non-zero"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Empty storage stays empty w/o charging_0"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'lasts', 'this', 'anyprev'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Empty storage stays empty w/o charging_1"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'lasts', 'this', 'anyprev'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Full storage stays full w/o discharging_0"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'lasts', 'this', 'anyprev'}
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:42 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Full storage stays full w/o discharging_1"
    > 11:02:42 - DEBUG - symenergy.auxiliary.constrcomb - {'lasts', 'this', 'anyprev'}
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Not full storage can't become full w/out charging"
    > 11:02:43 - DEBUG - symenergy.auxiliary.constrcomb - {'last', 'this'}
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Not empty storage can't become empty w/out discharging"
    > 11:02:43 - DEBUG - symenergy.auxiliary.constrcomb - {'last', 'this'}
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Empty storage can`t discharge"
    > 11:02:43 - DEBUG - symenergy.auxiliary.constrcomb - {'last', 'this'}
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All energy zero -> each charging cannot be non-zero"
    > 11:02:43 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All energy zero -> each discharging cannot be non-zero"
    > 11:02:43 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 0 column combinations: []
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All energy non-zero"
    > 11:02:43 - DEBUG - symenergy.auxiliary.constrcomb - {'all'}
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 1 column combinations: [(('act_lb_phs_pos_e_none', False),)]
    > 11:02:43 - INFO - symenergy.core.component - Generating constraint combinations for "curt"
    > 11:02:43 - INFO - symenergy.core.model - Length of merged df_comb: 5760
    > 11:02:43 - INFO - symenergy.core.model - ******************************model filtering******************************
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_n_pos_p_day', False), ('act_lb_curt_pos_p_day', False))
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 1920 (33.3%), remaining: 3840
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_n_pos_p_night', False), ('act_lb_curt_pos_p_night', False))
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 1280 (22.2%), remaining: 2560
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_g_pos_p_day', False), ('act_lb_curt_pos_p_day', False))
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 320 (5.6%), remaining: 2240
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_g_pos_p_night', False), ('act_lb_curt_pos_p_night', False))
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 280 (4.9%), remaining: 1960
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_phs_pos_pdch_night', False), ('act_lb_curt_pos_p_night', False))
    > 11:02:43 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 224 (3.9%), remaining: 1736
    > 11:02:43 - INFO - symenergy.core.model - Remaining df_comb rows: 1736
    > 11:02:43 - INFO - symenergy.core.model - Defining lagrangians...
    > 11:02:43 - DEBUG - symenergy.auxiliary.parallelization - ('NTHREADS: ', 'default')
    > 11:02:45 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 719/1736 (41.4%), chunksize 124, ForkPoolWorker-4
    > 11:02:45 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 789/1736 (45.4%), chunksize 124, ForkPoolWorker-1
    > 11:02:45 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 808/1736 (46.5%), chunksize 124, ForkPoolWorker-6
    > 11:02:45 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 808/1736 (46.5%), chunksize 124, ForkPoolWorker-5
    > 11:02:45 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 835/1736 (48.1%), chunksize 124, ForkPoolWorker-2
    > 11:02:45 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 844/1736 (48.6%), chunksize 124, ForkPoolWorker-3
    > 11:02:46 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1156/1736 (66.6%), chunksize 124, ForkPoolWorker-7
    > 11:02:46 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1552/1736 (89.4%), chunksize 124, ForkPoolWorker-1
    > 11:02:47 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1584/1736 (91.2%), chunksize 124, ForkPoolWorker-5
    > 11:02:47 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1596/1736 (91.9%), chunksize 124, ForkPoolWorker-6
    > 11:02:47 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1623/1736 (93.5%), chunksize 124, ForkPoolWorker-4
    > 11:02:47 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1634/1736 (94.1%), chunksize 124, ForkPoolWorker-2
    > 11:02:47 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1721/1736 (99.1%), chunksize 124, ForkPoolWorker-3
    > 11:02:47 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 1736/1736 (100.0%), chunksize 124, ForkPoolWorker-7
    > 11:02:48 - INFO - symenergy.auxiliary.parallelization - parallelize_df: chaining ... 
    > 11:02:48 - INFO - symenergy.auxiliary.parallelization - done.
    > 11:02:48 - INFO - symenergy.core.model - Getting selected variables/multipliers...
    > 11:02:48 - DEBUG - symenergy.auxiliary.parallelization - ('NTHREADS: ', 'default')
    > 11:02:56 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 776/1736 (44.7%), chunksize 124, ForkPoolWorker-8
    > 11:02:56 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 831/1736 (47.9%), chunksize 124, ForkPoolWorker-11
    > 11:02:56 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 854/1736 (49.2%), chunksize 124, ForkPoolWorker-14
    > 11:02:56 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 855/1736 (49.3%), chunksize 124, ForkPoolWorker-12
    > 11:02:56 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 857/1736 (49.4%), chunksize 124, ForkPoolWorker-10
    > 11:02:56 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 863/1736 (49.7%), chunksize 124, ForkPoolWorker-13
    > 11:02:57 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 868/1736 (50.0%), chunksize 124, ForkPoolWorker-9
    > 11:03:04 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 1682/1736 (96.9%), chunksize 124, ForkPoolWorker-8
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 1711/1736 (98.6%), chunksize 124, ForkPoolWorker-11
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 1726/1736 (99.4%), chunksize 124, ForkPoolWorker-12
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 1729/1736 (99.6%), chunksize 124, ForkPoolWorker-13
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 1730/1736 (99.7%), chunksize 124, ForkPoolWorker-9
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 1730/1736 (99.7%), chunksize 124, ForkPoolWorker-10
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 1736/1736 (100.0%), chunksize 124, ForkPoolWorker-14
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 11:03:05 - INFO - symenergy.auxiliary.parallelization - done.
    > 11:03:05 - INFO - symenergy.core.model - Solving
    > 11:03:05 - DEBUG - symenergy.auxiliary.parallelization - ('NTHREADS: ', 'default')
    > 11:05:19 - INFO - symenergy.auxiliary.parallelization - Solve: 824/1736 (47.5%), chunksize 124, ForkPoolWorker-17
    > 11:05:20 - INFO - symenergy.auxiliary.parallelization - Solve: 826/1736 (47.6%), chunksize 124, ForkPoolWorker-19
    > 11:05:27 - INFO - symenergy.auxiliary.parallelization - Solve: 861/1736 (49.6%), chunksize 124, ForkPoolWorker-16
    > 11:05:27 - INFO - symenergy.auxiliary.parallelization - Solve: 863/1736 (49.7%), chunksize 124, ForkPoolWorker-15
    > 11:05:29 - INFO - symenergy.auxiliary.parallelization - Solve: 868/1736 (50.0%), chunksize 124, ForkPoolWorker-20
    > 11:05:30 - INFO - symenergy.auxiliary.parallelization - Solve: 871/1736 (50.2%), chunksize 124, ForkPoolWorker-18
    > 11:05:39 - INFO - symenergy.auxiliary.parallelization - Solve: 922/1736 (53.1%), chunksize 124, ForkPoolWorker-21
    > 11:07:52 - INFO - symenergy.auxiliary.parallelization - Solve: 1684/1736 (97.0%), chunksize 124, ForkPoolWorker-15
    > 11:07:54 - INFO - symenergy.auxiliary.parallelization - Solve: 1689/1736 (97.3%), chunksize 124, ForkPoolWorker-17
    > 11:07:58 - INFO - symenergy.auxiliary.parallelization - Solve: 1712/1736 (98.6%), chunksize 124, ForkPoolWorker-16
    > 11:07:59 - INFO - symenergy.auxiliary.parallelization - Solve: 1720/1736 (99.1%), chunksize 124, ForkPoolWorker-19
    > 11:08:00 - INFO - symenergy.auxiliary.parallelization - Solve: 1721/1736 (99.1%), chunksize 124, ForkPoolWorker-18
    > 11:08:02 - INFO - symenergy.auxiliary.parallelization - Solve: 1729/1736 (99.6%), chunksize 124, ForkPoolWorker-21
    > 11:08:06 - INFO - symenergy.auxiliary.parallelization - Solve: 1736/1736 (100.0%), chunksize 124, ForkPoolWorker-20
    > 11:08:06 - INFO - symenergy.auxiliary.parallelization - parallelize_df: chaining ... 
    > 11:08:06 - INFO - symenergy.auxiliary.parallelization - done.
    > 11:08:06 - INFO - symenergy.core.model - Number of empty solutions: 1665 (95.9%)
    > 11:08:07 - WARNING - symenergy.core.model - Number of solutions with linear dependencies: Key 1: 40 (56.3%), Key 2: 0 (0.0%), Key 3: 0 (0.0%)
    > 11:08:07 - DEBUG - symenergy.auxiliary.parallelization - ('NTHREADS: ', 'default')
    > 11:08:07 - DEBUG - symenergy.core.model - idx=73
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=347
    > 11:08:07 - DEBUG - symenergy.core.model - idx=447
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=139
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=813
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=555
    > 11:08:07 - DEBUG - symenergy.core.model - idx=485
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=605
    > 11:08:07 - DEBUG - symenergy.core.model - idx=375
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=159
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=93
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model - idx=823
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:07 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 22/71 (31.0%), chunksize 5, ForkPoolWorker-24
    > 11:08:07 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=567
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=527
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=179
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=113
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=627
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=889
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 32/71 (45.1%), chunksize 5, ForkPoolWorker-26
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 32/71 (45.1%), chunksize 5, ForkPoolWorker-25
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=879
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=967
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 41/71 (57.7%), chunksize 5, ForkPoolWorker-28
    > 11:08:08 - DEBUG - symenergy.core.model - idx=945
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=123
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1167
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=189
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=665
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 45/71 (63.4%), chunksize 6, ForkPoolWorker-22
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 45/71 (63.4%), chunksize 5, ForkPoolWorker-27
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=977
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 47/71 (66.2%), chunksize 5, ForkPoolWorker-23
    > 11:08:08 - DEBUG - symenergy.core.model - idx=955
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1307
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1237
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 55/71 (77.5%), chunksize 5, ForkPoolWorker-24
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1467
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1517
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1617
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1011
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 58/71 (81.7%), chunksize 5, ForkPoolWorker-25
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1377
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1021
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1447
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1657
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1537
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model - idx=1477
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 67/71 (94.4%), chunksize 5, ForkPoolWorker-26
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 67/71 (94.4%), chunksize 5, ForkPoolWorker-28
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 68/71 (95.8%), chunksize 5, ForkPoolWorker-23
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:08 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:08 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 71/71 (100.0%), chunksize 5, ForkPoolWorker-27
    > 11:08:09 - DEBUG - symenergy.core.model - idx=1587
    > 11:08:09 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_none contained variabs pi_phs_pwrerg_chg_none, pi_phs_pwrerg_dch_none.
    > 11:08:09 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:09 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:09 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_chg_none contained variabs pi_phs_pwrerg_chg_none.
    > 11:08:09 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_dch_none contained variabs pi_phs_pwrerg_dch_none.
    > 11:08:09 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 71/71 (100.0%), chunksize 5, ForkPoolWorker-22
    > 11:08:09 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 11:08:09 - INFO - symenergy.auxiliary.parallelization - done.
    > 11:08:09 - INFO - symenergy.core.model - Generating total cost expressions...
    > 11:08:09 - DEBUG - symenergy.auxiliary.parallelization - ('NTHREADS: ', 'default')
    > 11:08:09 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 20/71 (28.2%), chunksize 5, ForkPoolWorker-30
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 24/71 (33.8%), chunksize 6, ForkPoolWorker-29
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 25/71 (35.2%), chunksize 5, ForkPoolWorker-32
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 40/71 (56.3%), chunksize 5, ForkPoolWorker-30
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 41/71 (57.7%), chunksize 5, ForkPoolWorker-33
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 43/71 (60.6%), chunksize 5, ForkPoolWorker-34
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 46/71 (64.8%), chunksize 5, ForkPoolWorker-31
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 50/71 (70.4%), chunksize 5, ForkPoolWorker-35
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 55/71 (77.5%), chunksize 5, ForkPoolWorker-29
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 63/71 (88.7%), chunksize 5, ForkPoolWorker-30
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 66/71 (93.0%), chunksize 5, ForkPoolWorker-32
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 67/71 (94.4%), chunksize 5, ForkPoolWorker-34
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 69/71 (97.2%), chunksize 5, ForkPoolWorker-31
    > 11:08:10 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 71/71 (100.0%), chunksize 5, ForkPoolWorker-33
    > 11:08:11 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 11:08:11 - INFO - symenergy.auxiliary.parallelization - done.


Adjust model parameters
-----------------------

Model parameters can be freely adjusted prior to the
:class:``sympy.Evaluator`` initialization.

.. code:: ipython3

    vre_day = 1
    vre_night = 0.1
    tot_l = m.slots['day'].l.value + m.slots['night'].l.value
    
    m.slots['day'].vre.value = vre_day / (vre_day + vre_night) * tot_l
    m.slots['night'].vre.value = vre_night / (vre_day + vre_night) * tot_l
    
    dd_75 = 14  # discharge duration for long-term storage
    dd_90 = 4   # discharge duration for short-term storage
    
    phs_C_max = m.slots['day'].l.value * 0.5  # maximum

Initialize evaluator instance, set model parameters and parameter sweep values.
-------------------------------------------------------------------------------

The model results are evaluated for all combinations of the iterables’
values in the ``x_vals`` argument.

The ``Evaluator.df_x_vals`` instance attribute is modified after
initialization to \* filter model runs \* set the storage energy
capacity ``E_phs`` in dependence on the efficiency ``eff_phs`` (i.e. the
storage type)

.. code:: ipython3

    x_vals_vre = {m.vre_scale: np.linspace(0, 1, 21),
                  m.comps['phs'].C: [0, 2600],
                  m.comps['phs'].E: [None], # <-- set later
                  m.comps['phs'].eff: [0.75, 0.9],
                  m.comps['n'].C: [2000, 4000, 5000],
                  }
    
    ev = evaluator.Evaluator(m, x_vals_vre, drop_non_optimum=False)
    ev.cache_lambd.delete()


.. parsed-literal::

    > 10:38:50 - WARNING - symenergy.auxiliary.io - **********************************************************************
    > 10:38:50 - WARNING - symenergy.auxiliary.io - **********************************************************************
    > 10:38:50 - WARNING - symenergy.auxiliary.io - Loading from cache file symenergy/cache/A44997457564.pickle.
    > 10:38:50 - WARNING - symenergy.auxiliary.io - Please delete this file to re-evaluate: Evaluator.cache_lambd.delete()
    > 10:38:50 - WARNING - symenergy.auxiliary.io - **********************************************************************
    > 10:38:50 - WARNING - symenergy.auxiliary.io - **********************************************************************


Adjust values for model evaluation
----------------------------------

The ``df_x_vals`` attribute is a dataframe containing all parameter
value combinations for which the model is to be evaluated. It can be
modified prior to the call

.. code:: ipython3

    ev.df_x_vals['E_phs_none'] = ev.df_x_vals.C_phs_none
    ev.df_x_vals.loc[ev.df_x_vals.eff_phs_none == 0.90, 'E_phs_none'] *= dd_90/12
    ev.df_x_vals.loc[ev.df_x_vals.eff_phs_none == 0.75, 'E_phs_none'] *= dd_75/12
    
    mask_vre = (ev.df_x_vals.vre_scale_none.isin(np.linspace(0, 1, 11))
                | ev.df_x_vals.vre_scale_none.isin(np.linspace(0.55, 0.75, 21)))
    mask_phs = (ev.df_x_vals.C_phs_none.isin(np.linspace(0, phs_C_max, 6)))
    
    ev.df_x_vals = ev.df_x_vals.loc[mask_vre | mask_phs]
    
    print(ev.df_x_vals.head(10))
    print('Length: ', len(ev.df_x_vals))


.. parsed-literal::

       vre_scale_none  C_phs_none   E_phs_none  eff_phs_none  C_n_none
    0             0.0           0     0.000000          0.75      2000
    1             0.0           0     0.000000          0.75      4000
    2             0.0           0     0.000000          0.75      5000
    3             0.0           0     0.000000          0.90      2000
    4             0.0           0     0.000000          0.90      4000
    5             0.0           0     0.000000          0.90      5000
    6             0.0        2600  3033.333333          0.75      2000
    7             0.0        2600  3033.333333          0.75      4000
    8             0.0        2600  3033.333333          0.75      5000
    9             0.0        2600   866.666667          0.90      2000
    Length:  210


Evaluate results for all entries of the ``Evaluator.df_x_vals table``
---------------------------------------------------------------------

**Note: Depending on the size of the model and the ``df_x_vals`` table
this takes a while.**

.. code:: ipython3

    ev.cache_eval.delete()
    ev.expand_to_x_vals_parallel()


.. parsed-literal::

    > 10:38:58 - WARNING - symenergy.evaluator.evaluator - _call_eval: Generating dataframe with length 193830


Add additional columns to the ``Evaluator.df_exp`` table
--------------------------------------------------------

Variables are not indexed by time slot names. The
``Evaluator.map_func_to_slot`` method expands the ``Evaluator.df_exp``
table by additional columns with variable names and time slots names.

.. code:: ipython3

    ev.map_func_to_slot()


::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-8-f8cbaa52f28e> in <module>
    ----> 1 ev.map_func_to_slot()
    

    AttributeError: 'Evaluator' object has no attribute 'map_func_to_slot'


Build supply table ``Evaluator.df_bal``
---------------------------------------

This includes the demand to the result demand and adjusts the signs,
such that demand, charging, and curtailment are negative.

.. code:: ipython3

    ev.build_supply_table()
    
    print(ev.df_bal.head(5))


.. parsed-literal::

           C_n_none  C_phs_none  E_phs_none  eff_phs_none        func  \
    69822      2000           0         0.0          0.75  curt_p_day   
    69834      2000           0         0.0          0.75  curt_p_day   
    69846      2000           0         0.0          0.75  curt_p_day   
    69858      2000           0         0.0          0.75  curt_p_day   
    69870      2000           0         0.0          0.75  curt_p_day   
    
          func_no_slot  idx     lambd pwrerg slot  vre_scale_none  
    69822       curt_p  627 -0.250000    pwr  day            0.55  
    69834       curt_p  627 -0.681818    pwr  day            0.60  
    69846       curt_p  627 -1.113636    pwr  day            0.65  
    69858       curt_p  627 -1.545455    pwr  day            0.70  
    69870       curt_p  627 -1.977273    pwr  day            0.75  


.. code:: ipython3

    from bokeh.io import show
    from bokeh.plotting import output_notebook
    import symenergy.evaluator.plotting as plotting
    
    output_notebook()
    
    balplot = plotting.BalancePlot(ev=ev, ind_axx='vre_scale_none', ind_pltx='slot', ind_plty=None)
    
    balplot.cds_pos.data
    
    show(balplot._get_layout())




.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="3227">Loading BokehJS ...</span>
        </div>





.. raw:: html

    
    
    
    
    
    
      <div class="bk-root" id="a014a632-a686-4332-80e8-03edcf63211c" data-root-id="3721"></div>





Simple energy balance plot with and without storage for day and night
---------------------------------------------------------------------

.. code:: ipython3

    df = ev.df_bal
    df = df.loc[-df.func_no_slot.str.contains('tc', 'lam')
               & df.eff_phs_none.isin([0.75])
               & df.C_n_none.isin([5000])
               & -df.slot.isin(['global'])]
    
    df['lambd'] = df.lambd.astype(float)
    df['vre_scale_none'] = df.vre_scale_none.apply(lambda x: round(x*10000)/10000)
    
    
    dfpv = df.pivot_table(columns='func_no_slot', values='lambd', index=['C_phs_none', 'slot', 'vre_scale_none'])
    
    list_slot = dfpv.index.get_level_values('slot').unique()
    list_c_phs = dfpv.index.get_level_values('C_phs_none').unique()
    
    fig, axarr = plt.subplots(len(list_c_phs),
                              len(list_slot), sharey=True, gridspec_kw={'wspace': 0.1,}, figsize=(15,15))
    
    for nslot, slot in enumerate(list_slot):
        for nc_phs, c_phs in enumerate(list_c_phs):
    
            ax = axarr[nslot][nc_phs]
            dfpv.loc[(c_phs, slot)].plot.bar(ax=ax, legend=False, use_index=True, stacked=True, width=1)
            
            ax.set_title('C_phs=%s, %s'%(c_phs, slot))
    
            
    leg = ax.legend(ncol=3)        
    




.. image:: example_constant_files/example_constant_18_0.png


Impact of storage on baseload production by constraint combination
------------------------------------------------------------------

.. code:: ipython3

    df = ev.df_exp
    df = df.loc[df.func.str.contains('n_p_')
               & df.is_optimum.isin([True])
               & -df.slot.isin(['global'])]
    
    df.head(5)
    
    dfdiff = df.pivot_table(index=[x for x in ev.x_name if not x in ['E_phs_none', 'C_phs_none']] + ['func'],
                            values='lambd', columns='C_phs_none')
    dfdiff['diff'] = dfdiff[2600] - dfdiff[0]
    #
    #dfcc = df.loc[df.C_phs_none == 2600].set_index(dfdiff.index.names)['idx']
    #dfdiff = dfdiff.reset_index().join(dfcc, on=dfdiff.index.names)
    
    dfdiff




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th></th>
          <th>C_phs_none</th>
          <th>0</th>
          <th>2600</th>
          <th>diff</th>
        </tr>
        <tr>
          <th>vre_scale_none</th>
          <th>eff_phs_none</th>
          <th>C_n_none</th>
          <th>func</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">0.0</th>
          <th rowspan="5" valign="top">0.75</th>
          <th rowspan="2" valign="top">2000</th>
          <th>n_p_day</th>
          <td>4.500000</td>
          <td>4.500000</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>n_p_night</th>
          <td>5.000000</td>
          <td>5.000000</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">4000</th>
          <th>n_p_day</th>
          <td>4.500000</td>
          <td>4.500000</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>n_p_night</th>
          <td>5.000000</td>
          <td>5.000000</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>5000</th>
          <th>n_p_day</th>
          <td>4.500000</td>
          <td>4.500000</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>...</th>
          <th>...</th>
          <th>...</th>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th rowspan="5" valign="top">1.0</th>
          <th rowspan="5" valign="top">0.90</th>
          <th>2000</th>
          <th>n_p_night</th>
          <td>4.136364</td>
          <td>4.136364</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">4000</th>
          <th>n_p_day</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>n_p_night</th>
          <td>4.136364</td>
          <td>4.136364</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">5000</th>
          <th>n_p_day</th>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>n_p_night</th>
          <td>4.136364</td>
          <td>4.136364</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    <p>252 rows × 3 columns</p>
    </div>



.. code:: ipython3

    #dfpv = dfdiff.pivot_table(index=['eff_phs_none', 'C_n_none', 'vre_scale_none'], 
    #                          columns='idx', values='diff')
    #
    #list_eff = dfpv.index.get_level_values('eff_phs_none').unique()
    #list_c_n = dfpv.index.get_level_values('C_n_none').unique()
    #
    #fig, axarr = plt.subplots(len(list_eff), len(list_c_n), 
    #                          sharey=True, gridspec_kw={'wspace': 0.1,}, figsize=(15,15))
    #
    #for neff, eff in enumerate(list_eff):
    #    for nc_n, c_n in enumerate(list_c_n):
    #
    #        ax = axarr[neff][nc_n] if isinstance(axarr, np.ndarray) else axarr
    #        dfpv.loc[(eff, c_n)].plot(ax=ax, legend=False, marker='.',use_index=True, stacked=False, linewidth=1)
    #        
    #        ax.set_title('C_phs_none=%s, %s'%(eff, c_n))
    #        ax.set_ylabel('Storage impact')
