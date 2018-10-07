#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic plotting methods.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import numpy as np
import pyAndy.core.plotpage as pltpg

class EvPlotting():
    '''
    Used as mixin class in evaluator.Evaluator.
    '''
    def line_plot(self, all_labels=False):

        data_kw = dict(ind_axx=[self.select_x.name], ind_pltx=['func_no_slot'],
                       ind_plty=['slot'],
                           series=['const_comb'], values=['lambd'],
                       aggfunc=np.mean, harmonize=True)
        page_kw = dict(left=0.05, right=0.99, bottom=0.050, top=0.8)
        plot_kw = dict(kind_def='LinePlot', stacked=False, on_values=True,
                       sharex=True, sharey=False, linewidth=4, marker=None,
                       xlabel=self.select_x.name, legend='')

        do = pltpg.PlotPageData.from_df(df=self.df_exp, **data_kw)
        plt0 = pltpg.PlotTiled(do, **plot_kw, **page_kw)

        #do_tc = do.copy()
        #do_tc.data = do_tc.data.loc[do_tc.data.index.get_level_values(data_kw['ind_pltx'][0]) == 'n_C_ret_None_lam_plot']
        #plttc = pltpg.PlotTiled(do_tc, **plot_kw, **page_kw)

        for ix, namex, iy, namey, plot, ax, kind in plt0.get_plot_ax_list():
            ''''''
            name_lin = ('lambd', 'cost_optimum')

            if name_lin in plot.linedict.keys():

                lin = plot.linedict[name_lin]
                lin.set_marker('x')
                lin.set_linewidth(0)

        lgdplotkey = (('tc_lam_plot',), ('global',), 'LinePlot')
        lgdplot = plt0.plotdict[lgdplotkey]
        hdl, lbl = lgdplot.ax.get_legend_handles_labels()

        if not all_labels:
            hdl, lbl = zip(*[(hh, ll) for hh, ll in zip(hdl, lbl)
                             if any(series_opt in ll
                                    for series_opt in self.const_comb_opt)])

        plt0.legend = 'page'

        plt0.add_page_legend(lgdplotkey, hdl, lbl)
