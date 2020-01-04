
.. code:: ipython3

    from plotly.offline import init_notebook_mode, iplot
    from plotly.graph_objs import *
    
    init_notebook_mode(connected=True)         # initiate notebook for offline plot
    
    trace0 = Scatter(
      x=[1, 2, 3, 4],
      y=[10, 15, 13, 17]
    )
    trace1 = Scatter(
      x=[1, 2, 3, 4],
      y=[16, 5, 11, 9]
    )
    data = Data([trace0, trace1])
    
    iplot(data)               # use plotly.offline.iplot for offline plot



.. raw:: html

            <script type="text/javascript">
            window.PlotlyConfig = {MathJaxConfig: 'local'};
            if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
            if (typeof require !== 'undefined') {
            require.undef("plotly");
            requirejs.config({
                paths: {
                    'plotly': ['https://cdn.plot.ly/plotly-latest.min']
                }
            });
            require(['plotly'], function(Plotly) {
                window._Plotly = Plotly;
            });
            }
            </script>
            


.. parsed-literal::

    /mnt/data/miniconda/envs/symenergy_new/lib/python3.6/site-packages/plotly/graph_objs/_deprecations.py:40: DeprecationWarning:
    
    plotly.graph_objs.Data is deprecated.
    Please replace it with a list or tuple of instances of the following types
      - plotly.graph_objs.Scatter
      - plotly.graph_objs.Bar
      - plotly.graph_objs.Area
      - plotly.graph_objs.Histogram
      - etc.
    
    



.. raw:: html

    <div>
            
            
                <div id="45920ab6-d348-423c-a6ee-82857ab42337" class="plotly-graph-div" style="height:525px; width:100%;"></div>
                <script type="text/javascript">
                    require(["plotly"], function(Plotly) {
                        window.PLOTLYENV=window.PLOTLYENV || {};
                        
                    if (document.getElementById("45920ab6-d348-423c-a6ee-82857ab42337")) {
                        Plotly.newPlot(
                            '45920ab6-d348-423c-a6ee-82857ab42337',
                            [{"type": "scatter", "x": [1, 2, 3, 4], "y": [10, 15, 13, 17]}, {"type": "scatter", "x": [1, 2, 3, 4], "y": [16, 5, 11, 9]}],
                            {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}},
                            {"responsive": true}
                        ).then(function(){
                                
    var gd = document.getElementById('45920ab6-d348-423c-a6ee-82857ab42337');
    var x = new MutationObserver(function (mutations, observer) {{
            var display = window.getComputedStyle(gd).display;
            if (!display || display === 'none') {{
                console.log([gd, 'removed!']);
                Plotly.purge(gd);
                observer.disconnect();
            }}
    }});
    
    // Listen for the removal of the full notebook cells
    var notebookContainer = gd.closest('#notebook-container');
    if (notebookContainer) {{
        x.observe(notebookContainer, {childList: true});
    }}
    
    // Listen for the clearing of the current output cell
    var outputEl = gd.closest('.output');
    if (outputEl) {{
        x.observe(outputEl, {childList: true});
    }}
    
                            })
                    };
                    });
                </script>
            </div>


Cookbook
========

.. code:: ipython3

    from symenergy.core import model


.. parsed-literal::

    > 12:16:50 - WARNING - symenergy.core.model - !!! Monkey-patching sympy.linsolve !!!


Simple example
--------------

We set up a simple model with two time slots, gas power plants with a
linear cost supply curve, and pumped-hydro storage (PHS) plants.

.. code:: ipython3

    m = model.Model(curtailment=True)
    
    m.add_slot(name='night', load=10, vre=1)
    m.add_slot(name='day', load=10, vre=1)
    
    m.add_storage(name='phs', eff=0.8)
    m.add_plant(name='gas', vc0=0, vc1=1)
    
    m.cache.delete_cached()
    m.generate_solve()


.. parsed-literal::

    > 12:47:05 - INFO - symenergy.core.asset - Variable pchg has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable pdch has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable e has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable pchg has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable pdch has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable e has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.slot - Generating time slot hash.
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.slot - Generating time slot hash.
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 12:47:05 - DEBUG - symenergy.assets.storage - Generating storage hash.
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 12:47:05 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 12:47:05 - INFO - symenergy.core.asset - Variable p has time dependence True
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.slot - Generating time slot hash.
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.slot - Generating time slot hash.
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 12:47:05 - DEBUG - symenergy.assets.storage - Generating storage hash.
    > 12:47:05 - DEBUG - symenergy.core.component - Generating component hash.
    > 12:47:05 - DEBUG - symenergy.core.asset - Generating asset hash.
    > 12:47:05 - INFO - symenergy.auxiliary.io - File doesn't exist. Could not remove symenergy/cache/2E0674EFD2F7.pickle
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Power plant output not simult. max end zero"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('p_cap_C',)
    > 12:47:06 - INFO - symenergy.core.component - ******************************gas******************************
    > 12:47:06 - INFO - symenergy.core.component - Component gas: Generating df_comb with length 4...
    > 12:47:06 - INFO - symenergy.core.component - ...done.
    > 12:47:06 - INFO - symenergy.core.component - ******************************night******************************
    > 12:47:06 - INFO - symenergy.core.component - Component night: Generating df_comb with length 1...
    > 12:47:06 - INFO - symenergy.core.component - ...done.
    > 12:47:06 - INFO - symenergy.core.component - ******************************day******************************
    > 12:47:06 - INFO - symenergy.core.component - Component day: Generating df_comb with length 1...
    > 12:47:06 - INFO - symenergy.core.component - ...done.
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Full storage can`t charge"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('e_cap_E',)
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "No simultaneous non-zero charging and non-zero discharging"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [(('act_lb_phs_pos_pchg_night', False), ('act_lb_phs_pos_pdch_night', False)), (('act_lb_phs_pos_pchg_day', False), ('act_lb_phs_pos_pdch_day', False))]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "No simultaneous full-power charging and full-power discharging"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('pchg_cap_C', 'pdch_cap_C')
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Storage energy not simult. full and empty"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('e_cap_E',)
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Storage charging not simult. max end zero"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('pchg_cap_C',)
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Storage discharging not simult. max end zero"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('pdch_cap_C',)
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All charging zero -> each discharging cannot be non-zero"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_pdch_night', False)], [('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_pdch_day', False)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All discharging zero -> each charging cannot be non-zero"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_pchg_night', False)], [('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_pchg_day', False)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All charging zero -> each energy cannot be non-zero"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_e_night', False)], [('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_e_day', False)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All discharging zero -> each energy cannot be non-zero"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_e_night', False)], [('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_e_day', False)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Empty storage stays empty w/o charging_0"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'anyprev', 'lasts', 'this'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_e_night', False)], [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_e_day', False)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Empty storage stays empty w/o charging_1"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'anyprev', 'lasts', 'this'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pchg_night', False), ('act_lb_phs_pos_e_night', True)], [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pchg_day', False), ('act_lb_phs_pos_e_day', True)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Full storage stays full w/o discharging_0"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('e_cap_E', 'e_cap_E')
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Full storage stays full w/o discharging_1"
    > 12:47:06 - WARNING - symenergy.auxiliary.constrcomb - Aborting gen_col_combs: missing constraints ('e_cap_E', 'e_cap_E')
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "Empty storage can`t discharge"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'last'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pdch_night', False)], [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pdch_day', False)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All energy zero -> each charging cannot be non-zero"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pchg_night', False)], [('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pchg_day', False)]]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Init CstrCombBase "All energy zero -> each discharging cannot be non-zero"
    > 12:47:06 - DEBUG - symenergy.auxiliary.constrcomb - {'this', 'all'}
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... expanded to 2 column combinations: [[('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pdch_night', False)], [('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pdch_day', False)]]
    > 12:47:06 - INFO - symenergy.core.component - ******************************phs******************************
    > 12:47:06 - INFO - symenergy.core.component - Component phs: Generating df_comb with length 64...
    > 12:47:06 - INFO - symenergy.core.component - ...done.
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_phs_pos_pchg_night', False), ('act_lb_phs_pos_pdch_night', False))
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 16 (25.0%), remaining: 48
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_phs_pos_pchg_day', False), ('act_lb_phs_pos_pdch_day', False))
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 12 (18.8%), remaining: 36
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_pdch_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 8 (12.5%), remaining: 28
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_pdch_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 4 (6.2%), remaining: 24
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_pchg_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 8 (12.5%), remaining: 16
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_pchg_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 4 (6.2%), remaining: 12
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_e_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 2 (3.1%), remaining: 10
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_e_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 1 (1.6%), remaining: 9
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_e_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 9
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_pdch_day', True), ('act_lb_phs_pos_pdch_night', True), ('act_lb_phs_pos_e_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 9
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pchg_night', True), ('act_lb_phs_pos_e_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 1 (1.6%), remaining: 8
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pchg_day', True), ('act_lb_phs_pos_e_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 1 (1.6%), remaining: 7
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pchg_night', False), ('act_lb_phs_pos_e_night', True)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 1 (1.6%), remaining: 6
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pchg_day', False), ('act_lb_phs_pos_e_day', True)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 1 (1.6%), remaining: 5
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pdch_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 5
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pdch_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 5
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pchg_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 5
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pchg_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 5
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_pdch_night', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 5
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: [('act_lb_phs_pos_e_day', True), ('act_lb_phs_pos_e_night', True), ('act_lb_phs_pos_pdch_day', False)]
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 0 (0.0%), remaining: 5
    > 12:47:06 - INFO - symenergy.core.component - ******************************curt******************************
    > 12:47:06 - INFO - symenergy.core.component - Component curt: Generating df_comb with length 4...
    > 12:47:06 - INFO - symenergy.core.component - ...done.
    > 12:47:06 - INFO - symenergy.core.model - Length of merged df_comb: 80
    > 12:47:06 - INFO - symenergy.core.model - ******************************model filtering******************************
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_gas_pos_p_night', False), ('act_lb_curt_pos_p_night', False))
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 20 (25.0%), remaining: 60
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_gas_pos_p_day', False), ('act_lb_curt_pos_p_day', False))
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 15 (18.8%), remaining: 45
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_phs_pos_pdch_night', False), ('act_lb_curt_pos_p_night', False))
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 6 (7.5%), remaining: 39
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - Deleting constraint combination: (('act_lb_phs_pos_pdch_day', False), ('act_lb_curt_pos_p_day', False))
    > 12:47:06 - INFO - symenergy.auxiliary.constrcomb - ... total deleted: 6 (7.5%), remaining: 33
    > 12:47:06 - INFO - symenergy.core.model - Defining lagrangians...
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 18/33 (54.5%), chunksize 3, tavg=7.4ms, tcur=7.3ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 18/33 (54.5%), chunksize 2, tavg=11.0ms, tcur=10.7ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 10/33 (30.3%), chunksize 3, tavg=7.3ms, tcur=7.9ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 11/33 (33.3%), chunksize 3, tavg=7.3ms, tcur=7.4ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 19/33 (57.6%), chunksize 2, tavg=11.1ms, tcur=15.1ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 6/33 (18.2%), chunksize 3, tavg=7.3ms, tcur=7.3ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 12/33 (36.4%), chunksize 3, tavg=7.4ms, tcur=8.3ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 24/33 (72.7%), chunksize 2, tavg=11.0ms, tcur=9.9ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 23/33 (69.7%), chunksize 2, tavg=11.0ms, tcur=6.3ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 33/33 (100.0%), chunksize 2, tavg=11.0ms, tcur=8.5ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 27/33 (81.8%), chunksize 2, tavg=11.0ms, tcur=12.9ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 29/33 (87.9%), chunksize 2, tavg=11.1ms, tcur=14.9ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 31/33 (93.9%), chunksize 2, tavg=11.0ms, tcur=7.4ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Construct lagrange: 32/33 (97.0%), chunksize 2, tavg=11.1ms, tcur=13.4ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - done.
    > 12:47:07 - INFO - symenergy.core.model - Getting selected variables/multipliers...
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 18/33 (54.5%), chunksize 2, tavg=13.1ms, tcur=14.1ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 19/33 (57.6%), chunksize 2, tavg=13.1ms, tcur=14.8ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 21/33 (63.6%), chunksize 3, tavg=8.8ms, tcur=10.4ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 19/33 (57.6%), chunksize 3, tavg=8.7ms, tcur=10.2ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 19/33 (57.6%), chunksize 3, tavg=8.7ms, tcur=9.1ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 18/33 (54.5%), chunksize 3, tavg=8.7ms, tcur=8.7ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 19/33 (57.6%), chunksize 3, tavg=8.8ms, tcur=9.8ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 28/33 (84.8%), chunksize 2, tavg=13.2ms, tcur=13.8ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 29/33 (87.9%), chunksize 2, tavg=13.2ms, tcur=16.0ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 31/33 (93.9%), chunksize 2, tavg=13.2ms, tcur=13.9ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 33/33 (100.0%), chunksize 2, tavg=13.2ms, tcur=14.8ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 33/33 (100.0%), chunksize 2, tavg=13.3ms, tcur=16.4ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 33/33 (100.0%), chunksize 2, tavg=13.3ms, tcur=15.5ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - Get variabs/multipliers: 33/33 (100.0%), chunksize 2, tavg=13.4ms, tcur=21.6ms
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 12:47:07 - INFO - symenergy.auxiliary.parallelization - done.
    > 12:47:07 - INFO - symenergy.core.model - Solving
    > 12:47:10 - INFO - symenergy.auxiliary.parallelization - Solve: 12/33 (36.4%), chunksize 2, tavg=633.1ms, tcur=633.1ms
    > 12:47:10 - INFO - symenergy.auxiliary.parallelization - Solve: 12/33 (36.4%), chunksize 2, tavg=633.6ms, tcur=687.3ms
    > 12:47:11 - INFO - symenergy.auxiliary.parallelization - Solve: 18/33 (54.5%), chunksize 3, tavg=421.6ms, tcur=345.1ms
    > 12:47:11 - INFO - symenergy.auxiliary.parallelization - Solve: 21/33 (63.6%), chunksize 3, tavg=421.3ms, tcur=389.7ms
    > 12:47:11 - INFO - symenergy.auxiliary.parallelization - Solve: 21/33 (63.6%), chunksize 3, tavg=421.0ms, tcur=392.6ms
    > 12:47:11 - INFO - symenergy.auxiliary.parallelization - Solve: 23/33 (69.7%), chunksize 3, tavg=421.2ms, tcur=440.4ms
    > 12:47:12 - INFO - symenergy.auxiliary.parallelization - Solve: 25/33 (75.8%), chunksize 2, tavg=629.3ms, tcur=379.4ms
    > 12:47:12 - INFO - symenergy.auxiliary.parallelization - Solve: 25/33 (75.8%), chunksize 2, tavg=626.5ms, tcur=343.8ms
    > 12:47:12 - INFO - symenergy.auxiliary.parallelization - Solve: 32/33 (97.0%), chunksize 2, tavg=623.4ms, tcur=323.5ms
    > 12:47:13 - INFO - symenergy.auxiliary.parallelization - Solve: 33/33 (100.0%), chunksize 2, tavg=620.8ms, tcur=362.6ms
    > 12:47:13 - INFO - symenergy.auxiliary.parallelization - Solve: 33/33 (100.0%), chunksize 3, tavg=416.1ms, tcur=633.5ms
    > 12:47:13 - INFO - symenergy.auxiliary.parallelization - Solve: 33/33 (100.0%), chunksize 2, tavg=624.6ms, tcur=668.5ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Solve: 33/33 (100.0%), chunksize 2, tavg=623.1ms, tcur=483.4ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Solve: 33/33 (100.0%), chunksize 2, tavg=623.1ms, tcur=615.6ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - done.
    > 12:47:14 - INFO - symenergy.core.model - Number of empty solutions: 19 (57.6%)
    > 12:47:14 - WARNING - symenergy.core.model - Number of solutions with linear dependencies: Key 1: 4 (28.6%), Key 2: 0 (0.0%), Key 3: 0 (0.0%)
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 1/14 (7.1%), chunksize 1, tavg=23.7ms, tcur=23.7ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 2/14 (14.3%), chunksize 1, tavg=23.7ms, tcur=23.6ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 5/14 (35.7%), chunksize 1, tavg=23.7ms, tcur=18.1ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 6/14 (42.9%), chunksize 1, tavg=23.7ms, tcur=24.0ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 4/14 (28.6%), chunksize 1, tavg=23.7ms, tcur=21.4ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 9/14 (64.3%), chunksize 1, tavg=23.5ms, tcur=9.1ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 9/14 (64.3%), chunksize 1, tavg=23.4ms, tcur=7.8ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 10/14 (71.4%), chunksize 1, tavg=23.2ms, tcur=8.2ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 12/14 (85.7%), chunksize 1, tavg=23.0ms, tcur=7.7ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 13/14 (92.9%), chunksize 1, tavg=22.9ms, tcur=8.1ms
    > 12:47:14 - DEBUG - symenergy.core.model - idx=4
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_day contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_night contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model - idx=19
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_day contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model - idx=11
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_night contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_day contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_night contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model - idx=29
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_day contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 14/14 (100.0%), chunksize 1, tavg=25.2ms, tcur=251.6ms
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_e_night contained variabs pi_phs_pwrerg_day, pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 14/14 (100.0%), chunksize 1, tavg=26.5ms, tcur=158.0ms
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pchg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 14/14 (100.0%), chunksize 1, tavg=28.6ms, tcur=232.2ms
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for lb_phs_pos_pdch_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_day contained variabs pi_phs_pwrerg_day.
    > 12:47:14 - DEBUG - symenergy.core.model -      Solution for pi_phs_pwrerg_night contained variabs pi_phs_pwrerg_night.
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - Fix linear dependencies: 14/14 (100.0%), chunksize 1, tavg=29.6ms, tcur=136.4ms
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 12:47:14 - INFO - symenergy.auxiliary.parallelization - done.
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 7/14 (50.0%), chunksize 1, tavg=201.6ms, tcur=166.7ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 7/14 (50.0%), chunksize 1, tavg=202.0ms, tcur=202.0ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 9/14 (64.3%), chunksize 1, tavg=203.8ms, tcur=414.4ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 9/14 (64.3%), chunksize 1, tavg=208.3ms, tcur=421.6ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 9/14 (64.3%), chunksize 1, tavg=206.1ms, tcur=439.4ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 12/14 (85.7%), chunksize 1, tavg=206.7ms, tcur=56.4ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 12/14 (85.7%), chunksize 1, tavg=208.2ms, tcur=353.7ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 14/14 (100.0%), chunksize 1, tavg=208.1ms, tcur=194.6ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 14/14 (100.0%), chunksize 1, tavg=206.7ms, tcur=68.2ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 14/14 (100.0%), chunksize 1, tavg=205.0ms, tcur=41.5ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 14/14 (100.0%), chunksize 1, tavg=208.4ms, tcur=539.9ms
    > 12:47:15 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 14/14 (100.0%), chunksize 1, tavg=210.4ms, tcur=405.7ms
    > 12:47:16 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 14/14 (100.0%), chunksize 1, tavg=225.5ms, tcur=1724.3ms
    > 12:47:17 - INFO - symenergy.auxiliary.parallelization - Substituting total cost: 14/14 (100.0%), chunksize 1, tavg=243.3ms, tcur=2006.0ms
    > 12:47:17 - INFO - symenergy.auxiliary.parallelization - parallelize_df: concatenating ... 
    > 12:47:17 - INFO - symenergy.auxiliary.parallelization - done.


Investigate closed-form numerical solutions
-------------------------------------------

All results are stored in the model attribute ``df_comb``. This
``pandas.DataFrame`` is indexed by the inequality constraint
combinations. The column names corresponding to the active/inactive
constraints can be listed through the model’s ``constraints``
collection:

.. code:: ipython3

     m.constraints('col', is_equality_constraint=False)




.. parsed-literal::

    ['act_lb_gas_pos_p_night',
     'act_lb_gas_pos_p_day',
     'act_lb_phs_pos_pchg_night',
     'act_lb_phs_pos_pchg_day',
     'act_lb_phs_pos_pdch_night',
     'act_lb_phs_pos_pdch_day',
     'act_lb_phs_pos_e_night',
     'act_lb_phs_pos_e_day',
     'act_lb_curt_pos_p_night',
     'act_lb_curt_pos_p_day']



For example, to select the constraint combination where storage is
inactive and all power is supplied from gas plants, we would select all
columns where the positivity constraints of storage operation are active
(``True``) and positivity constraints of gas power plant operation are
inactive (``False``):

.. code:: ipython3

    df_slct = m.df_comb.set_index('idx').query('act_lb_phs_pos_pchg_night and act_lb_phs_pos_pchg_day'
                                               ' and act_lb_phs_pos_pdch_night and act_lb_phs_pos_pdch_day'
                                               ' and not act_lb_gas_pos_p_night and not act_lb_gas_pos_p_day')
    display(df_slct.T)



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
          <th>idx</th>
          <th>4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>act_lb_gas_pos_p_night</th>
          <td>False</td>
        </tr>
        <tr>
          <th>act_lb_gas_pos_p_day</th>
          <td>False</td>
        </tr>
        <tr>
          <th>act_lb_phs_pos_pchg_night</th>
          <td>True</td>
        </tr>
        <tr>
          <th>act_lb_phs_pos_pchg_day</th>
          <td>True</td>
        </tr>
        <tr>
          <th>act_lb_phs_pos_pdch_night</th>
          <td>True</td>
        </tr>
        <tr>
          <th>act_lb_phs_pos_pdch_day</th>
          <td>True</td>
        </tr>
        <tr>
          <th>act_lb_phs_pos_e_night</th>
          <td>True</td>
        </tr>
        <tr>
          <th>act_lb_phs_pos_e_day</th>
          <td>True</td>
        </tr>
        <tr>
          <th>act_lb_curt_pos_p_night</th>
          <td>True</td>
        </tr>
        <tr>
          <th>act_lb_curt_pos_p_day</th>
          <td>True</td>
        </tr>
        <tr>
          <th>lagrange</th>
          <td>curt_p_day*lb_curt_pos_p_day + curt_p_night*lb...</td>
        </tr>
        <tr>
          <th>variabs_multips</th>
          <td>[curt_p_day, curt_p_night, gas_p_day, gas_p_ni...</td>
        </tr>
        <tr>
          <th>result</th>
          <td>[0, 0, l_day - vre_day*vre_scale_none, l_night...</td>
        </tr>
        <tr>
          <th>code_lindep</th>
          <td>1</td>
        </tr>
        <tr>
          <th>tc</th>
          <td>w_none*(2*vc0_gas_none*(l_day - vre_day*vre_sc...</td>
        </tr>
      </tbody>
    </table>
    </div>


Not that the only valid solution also has zero curtailment (active
``act_lb_curt_pos_p_...`` constraints). This is because simultaneous
non-zero generator output and non-zero curtailment is excluded a-priori
throught the definition of mutually exclusive constraints in the model
class.

The filtered tablel above tells us the index of the relevant constraint
combination. The model class provides a convenience method
``print_results`` to print the corresponding closed-form solutions for a
given index.

.. code:: ipython3

    m.print_results(m.df_comb, idx=df_slct.index.tolist()[0])


.. parsed-literal::

    ******************** gas_p_day ********************
    l_day - vre_day*vre_scale_none
    ******************** gas_p_night ********************
    l_night - vre_night*vre_scale_none
    ******************** lb_phs_pos_e_day ********************
    0
    ******************** lb_phs_pos_e_night ********************
    0
    ******************** lb_phs_pos_pchg_day ********************
    0
    ******************** lb_phs_pos_pchg_night ********************
    0
    ******************** lb_phs_pos_pdch_day ********************
    0
    ******************** lb_phs_pos_pdch_night ********************
    0
    ******************** phs_e_day ********************
    0
    ******************** phs_e_night ********************
    0
    ******************** phs_pchg_day ********************
    0
    ******************** phs_pchg_night ********************
    0
    ******************** phs_pdch_day ********************
    0
    ******************** phs_pdch_night ********************
    0
    ******************** pi_phs_pwrerg_day ********************
    0
    ******************** pi_phs_pwrerg_night ********************
    0
    ******************** pi_supply_day ********************
    w_none*(vc0_gas_none + vc1_gas_none*(l_day - vre_day*vre_scale_none))
    ******************** pi_supply_night ********************
    w_none*(vc0_gas_none + vc1_gas_none*(l_night - vre_night*vre_scale_none))


As expected, all storage operation is zero (charging ``phs_pdch_day``,
``phs_pdch_night``, discharging ``phs_pdch_day``, ``phs_pdch_night``,
stored energy ``phs_e_day``, ``phs_e_night``). The gas power production
is used to cover the residual load during the day
``l_day - vre_day*vre_scale_none`` and at night
``l_night - vre_night*vre_scale_none``.

Numerical evaluation
--------------------

The :class:``symenergy.evaluator.evaluator.Evaluator`` allows to
evaluate the model results for certain parameter values

.. code:: ipython3

    import evaluator.evaluator as evaluator
    import numpy as np
    
    from evaluator.evaluator import logger
    logger.setLevel('INFO')
    
    x_vals = {
              m.vre_scale: np.linspace(0, 1.5, 31),
              m.slots['day'].vre: np.linspace(1, 9, 9),
              m.storages['phs'].eff: np.linspace(0.01, 0.99, 3),
             }
    
    ev = evaluator.Evaluator(m, x_vals)


.. parsed-literal::

    > 13:09:12 - INFO - evaluator.evaluator - Generating lambda functions for pi_supply_night.
    > 13:09:12 - INFO - evaluator.evaluator - Generating lambda functions for pi_supply_day.
    > 13:09:12 - INFO - evaluator.evaluator - Generating lambda functions for gas_p_night.
    > 13:09:13 - INFO - evaluator.evaluator - Generating lambda functions for gas_p_day.
    > 13:09:13 - INFO - evaluator.evaluator - Generating lambda functions for phs_pchg_night.
    > 13:09:13 - INFO - evaluator.evaluator - Generating lambda functions for phs_pchg_day.
    > 13:09:13 - INFO - evaluator.evaluator - Generating lambda functions for phs_pdch_night.
    > 13:09:14 - INFO - evaluator.evaluator - Generating lambda functions for phs_pdch_day.
    > 13:09:14 - INFO - evaluator.evaluator - Generating lambda functions for phs_e_night.
    > 13:09:14 - INFO - evaluator.evaluator - Generating lambda functions for phs_e_day.
    > 13:09:14 - INFO - evaluator.evaluator - Generating lambda functions for curt_p_night.
    > 13:09:14 - INFO - evaluator.evaluator - Generating lambda functions for curt_p_day.
    > 13:09:15 - INFO - evaluator.evaluator - Generating lambda functions for tc.


.. code:: ipython3

    ev.expand_to_x_vals()
    ev.build_supply_table()


.. parsed-literal::

    > 13:09:37 - INFO - evaluator.evaluator - 19.768859386444092


The ``evaluator.plotting`` module provides classes to generate
interactive Bokeh plots. Below we show the energy balance as a function
of the VRE scaling parameter ``'vre_scale_none'``, with horizontal
subplots by ``'slot'``. The remaining free parameters are added as lists
from which values can be selected interactively.

.. code:: ipython3

    import evaluator.plotting as plotting
    from bokeh.io import show, output_notebook
    output_notebook(verbose=False)
    
    balplot = plotting.BalancePlot(ev, ind_axx='vre_scale_none', ind_pltx='slot', ind_plty=None)
    show(balplot._get_layout())



.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="14263">Loading BokehJS ...</span>
        </div>





.. raw:: html

    
    
    
    
    
    
      <div class="bk-root" id="91504ba0-429a-47eb-9c48-3d67dfe64793" data-root-id="14706"></div>





The evaluator results can also be used to determine relevant constraint
combinations for further analysis. Based on the plots above, we might be
interested in the optimal solution corresponding to the parameter values
``vre_day == 9``, ``eff_phs_non == 0.5`` and ``vre_scale_none == 1``.

.. code:: ipython3

    df_slct = ev.df_exp.query('is_optimum'
                              ' and vre_day == 9'
                              ' and eff_phs_none == 0.5'
                              ' and vre_scale_none == 1')
    display(df_slct[ev.x_name + ['idx', 'func', 'lambd', 'is_optimum']])



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
          <th>vre_scale_none</th>
          <th>vre_day</th>
          <th>eff_phs_none</th>
          <th>idx</th>
          <th>func</th>
          <th>lambd</th>
          <th>is_optimum</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1132</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>curt_p_day_lam_plot</td>
          <td>0.000000</td>
          <td>True</td>
        </tr>
        <tr>
          <th>9070</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>curt_p_night_lam_plot</td>
          <td>0.000000</td>
          <td>True</td>
        </tr>
        <tr>
          <th>17008</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>gas_p_day_lam_plot</td>
          <td>3.799434</td>
          <td>True</td>
        </tr>
        <tr>
          <th>24946</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>gas_p_night_lam_plot</td>
          <td>7.600283</td>
          <td>True</td>
        </tr>
        <tr>
          <th>32884</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>phs_e_day_lam_plot</td>
          <td>1.979499</td>
          <td>True</td>
        </tr>
        <tr>
          <th>40822</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>phs_e_night_lam_plot</td>
          <td>0.000000</td>
          <td>True</td>
        </tr>
        <tr>
          <th>48760</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>phs_pchg_day_lam_plot</td>
          <td>2.799434</td>
          <td>True</td>
        </tr>
        <tr>
          <th>56698</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>phs_pchg_night_lam_plot</td>
          <td>0.000000</td>
          <td>True</td>
        </tr>
        <tr>
          <th>64636</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>phs_pdch_day_lam_plot</td>
          <td>0.000000</td>
          <td>True</td>
        </tr>
        <tr>
          <th>72574</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>phs_pdch_night_lam_plot</td>
          <td>1.399717</td>
          <td>True</td>
        </tr>
        <tr>
          <th>80512</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>pi_supply_day_lam_plot</td>
          <td>3.799434</td>
          <td>True</td>
        </tr>
        <tr>
          <th>88450</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>pi_supply_night_lam_plot</td>
          <td>7.600283</td>
          <td>True</td>
        </tr>
        <tr>
          <th>96388</th>
          <td>1.0</td>
          <td>9.0</td>
          <td>0.5</td>
          <td>3</td>
          <td>tc_lam_plot</td>
          <td>36.101980</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    </div>


The index of the corresponding constraint combination is thus 3.

Again, we can print the results using the ``Model.print_results``
function to obtain the corresponding closed-form symbolic solutions.

.. code:: ipython3

    m.print_results(m.df_comb, idx=df_slct.idx.tolist()[0])


.. parsed-literal::

    ******************** curt_p_day ********************
    0
    ******************** curt_p_night ********************
    0
    ******************** gas_p_day ********************
    eff_phs_none**(-0.5)*(-eff_phs_none**0.5*(0.001*eff_phs_none**0.5 + vc0_gas_none)*(eff_phs_none**2.0 + 1) + eff_phs_none**1.5*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)))/(vc1_gas_none*(eff_phs_none**2.0 + 1))
    ******************** gas_p_night ********************
    1.0*(1.0*eff_phs_none**1.0*l_day*vc1_gas_none + 1.0*eff_phs_none**1.0*vc0_gas_none - 1.0*eff_phs_none**1.0*vc1_gas_none*vre_day*vre_scale_none + 0.001*eff_phs_none**1.5 - 1.0*eff_phs_none**2.0*vc0_gas_none + 1.0*l_night*vc1_gas_none - 1.0*vc1_gas_none*vre_night*vre_scale_none)/(vc1_gas_none*(eff_phs_none**2.0 + 1))
    ******************** lb_curt_pos_p_day ********************
    w_none*(0.001*eff_phs_none**0.5*(eff_phs_none**2.0 + 1) - eff_phs_none**1.0*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)))/(eff_phs_none**2.0 + 1)
    ******************** lb_curt_pos_p_night ********************
    -w_none*(eff_phs_none**1.0*(vc0_gas_none + vc1_gas_none*(l_day - vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none + vc1_gas_none*(l_night - vre_night*vre_scale_none))/(eff_phs_none**2.0 + 1)
    ******************** lb_phs_pos_e_night ********************
    -0.00200000000000000
    ******************** lb_phs_pos_pchg_night ********************
    w_none*(eff_phs_none**1.0 - 1)*(eff_phs_none**1.0*(vc0_gas_none + vc1_gas_none*(l_day - vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none + vc1_gas_none*(l_night - vre_night*vre_scale_none))/(eff_phs_none**2.0 + 1)
    ******************** lb_phs_pos_pdch_day ********************
    eff_phs_none**(-1.5)*w_none*(1 - eff_phs_none**1.0)*(0.001*eff_phs_none**1.0*(eff_phs_none**2.0 + 1) - eff_phs_none**1.5*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)))/(eff_phs_none**2.0 + 1)
    ******************** phs_e_day ********************
    eff_phs_none**0.5*w_none*(eff_phs_none**1.0*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)) - (eff_phs_none**2.0 + 1)*(0.001*eff_phs_none**0.5 + vc0_gas_none + vc1_gas_none*(l_day - vre_day*vre_scale_none)))/(vc1_gas_none*(eff_phs_none**2.0 + 1))
    ******************** phs_e_night ********************
    0
    ******************** phs_pchg_day ********************
    eff_phs_none**(-0.5)*(-eff_phs_none**0.5*(eff_phs_none**2.0 + 1)*(0.001*eff_phs_none**0.5 + vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + eff_phs_none**1.5*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)))/(vc1_gas_none*(eff_phs_none**2.0 + 1))
    ******************** phs_pchg_night ********************
    0
    ******************** phs_pdch_day ********************
    0
    ******************** phs_pdch_night ********************
    eff_phs_none**1.0*(eff_phs_none**1.0*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)) - (eff_phs_none**2.0 + 1)*(0.001*eff_phs_none**0.5 + vc0_gas_none + vc1_gas_none*(l_day - vre_day*vre_scale_none)))/(vc1_gas_none*(eff_phs_none**2.0 + 1))
    ******************** pi_phs_pwrerg_day ********************
    eff_phs_none**(-0.5)*(0.001*eff_phs_none**0.5*(eff_phs_none**2.0 + 1) - eff_phs_none**1.0*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)))/(eff_phs_none**2.0 + 1)
    ******************** pi_phs_pwrerg_night ********************
    -eff_phs_none**0.5*(eff_phs_none**1.0*(vc0_gas_none + vc1_gas_none*(l_day - vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none + vc1_gas_none*(l_night - vre_night*vre_scale_none))/(eff_phs_none**2.0 + 1)
    ******************** pi_supply_day ********************
    w_none*(-0.001*eff_phs_none**0.5*(eff_phs_none**2.0 + 1) + eff_phs_none**1.0*(eff_phs_none**1.0*(vc0_gas_none - vc1_gas_none*(-l_day + vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none - vc1_gas_none*(-l_night + vre_night*vre_scale_none)))/(eff_phs_none**2.0 + 1)
    ******************** pi_supply_night ********************
    w_none*(eff_phs_none**1.0*(vc0_gas_none + vc1_gas_none*(l_day - vre_day*vre_scale_none)) + 0.001*eff_phs_none**1.5 + vc0_gas_none + vc1_gas_none*(l_night - vre_night*vre_scale_none))/(eff_phs_none**2.0 + 1)

