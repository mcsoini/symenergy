#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:30:13 2019

@author: user
"""

import itertools
from bokeh.models.sources import ColumnDataSource, CustomJS
from bokeh.models import Slider, CDSView, GroupFilter
from bokeh.layouts import column, row, gridplot, WidgetBox
from bokeh.models import Panel, Legend
from bokeh.plotting import figure, show
from bokeh.models.widgets import Tabs, MultiSelect, Button
from bokeh.core.properties import value
from bokeh.palettes import brewer
from bokeh.io import output_notebook, output_file

#output_notebook(verbose=False)
#output_file('symenergy_analysis.html')


class SymenergyPlotting():
    '''
    Mixin class for Evaluator.
    '''


    def plot_energy_balance(self, ind_pltx='slot',
                            ind_plty=None, ind_axx='vre_scale'):

        ind_pltx='slot'
        ind_plty=None
        ind_axx='vre_scale'

        df = self.df_bal
        df = df.loc[-df.func_no_slot.str.contains('tc', 'lam')
                   & -df.slot.isin(['global'])
                   & -df.pwrerg.isin(['erg'])
                   ]

        df['lambd'] = df.lambd.astype(float)

        df = df.sort_values(ind_axx)

        ind_all = list(filter(lambda x: x is not None, [ind_axx, ind_pltx, ind_plty]))
        ind_slct = [x for x in self.x_name if not x in ind_all]
        dfgp = df.pivot_table(index=ind_slct + ind_all, columns='func_no_slot',  values='lambd')
        dfgp.columns.names = [None]

        cols_to_str = {key: val for key, val in {ind_pltx: lambda x: x[ind_pltx].apply(str),
                       ind_plty: lambda x: x[ind_plty].apply(str)}.items() if key is not None}
        dfgp = dfgp.reset_index().assign(**cols_to_str).set_index(dfgp.index.names)

        slct_def = list(dfgp.iloc[[0]].index)[0][:len(ind_slct)]

        cols_neg = ['l', 'pchg', 'curt_p']
        cols_neg = [c for c in dfgp.columns if any(c.endswith(pat) for pat in cols_neg)]
        cols_pos = [c for c in dfgp.columns if c not in cols_neg]

        options_dict = {ind: dfgp.index.get_level_values(ind).unique().tolist()
                        for ind in dfgp.index.names}

        data_curr_pos=ColumnDataSource(dfgp[cols_pos].loc[slct_def].reset_index())
        data_overall_pos=ColumnDataSource(dfgp[cols_pos].reset_index())

        data_curr_neg=ColumnDataSource(dfgp[cols_neg].loc[slct_def].reset_index())
        data_overall_neg=ColumnDataSource(dfgp[cols_neg].reset_index())

        def get_view(source, valx, valy):
            filters = []
            if ind_pltx:
                 filters += ([GroupFilter(column_name=ind_pltx, group=valx)] if ind_pltx else [])
            if ind_plty:
                 filters += ([GroupFilter(column_name=ind_plty, group=valy)] if ind_plty else [])

            return CDSView(source=source, filters=filters)


        xy_combs = list(itertools.product(options_dict[ind_pltx] if ind_pltx else [None],
                                          options_dict[ind_plty] if ind_plty else [None]))
        views = {(valx, valy): {posneg: get_view(source, valx, valy)
                                for source, posneg
                                in zip([data_curr_pos, data_curr_neg], ['pos', 'neg'])}
                 for valx, valy in xy_combs}


        var_slct_str = '; '.join('var {inds} = slct_{inds}.value'.format(inds=inds)
                for inds in ind_slct) + ';'
        param_str = ', ' + ', '.join(ind_slct)
        match_str = ' && '.join('source.data[\'{inds}\'][i] == {inds}'.format(inds=inds) for inds in ind_slct)

        def get_push_str(posneg, plant, var):
            return ('sc_{pn}.data[\'{plant}{var}\']'
                    '.push(source_{pn}.data[\'{plant}{var}\'][i]); '
                   ).format(pn=posneg, plant=plant, var=var)

        push_str_pos = '; '.join([
            ''.join([get_push_str('pos', store_name, '_pdch') for store_name in self.model.storages]),
            ''.join([get_push_str('pos', plant_name, '_p') for plant_name in self.model.plants]),
            get_push_str('pos', 'vre', '')
        ])

        push_str_neg = '; '.join([
            get_push_str('neg', 'l', ''),
            get_push_str('neg', 'curt', '_p') if 'curt' in self.model.comps else '',
            ''.join([get_push_str('neg', store_name, '_pchg') for store_name in self.model.storages]),
        ])


        def get_init_str(list_p, list_var, list_pn):
            return ''.join('sc_%s.data[\'%s%s\'] = []; '%(pn, p, var)
                for p, var, pn
                in [(p, *vpn)for p, vpn in itertools.product(list_p, zip(list_var, list_pn))]
            )

        init_str = '; '.join([
            get_init_str(['l'], [''], ['neg']),
            get_init_str(['vre'], [''], ['pos']),
            get_init_str(self.model.storages, ['_pdch', '_pchg'], ['pos', 'neg']),
            get_init_str(self.model.plants, ['_p'], ['pos']),
            get_init_str(['curt'], ['_p'], ['neg']) if 'curt' in self.model.comps else '',
        ])

        callback = CustomJS(args=dict(source_pos=data_overall_pos,
                                      source_neg=data_overall_neg,
                                      sc_pos=data_curr_pos,
                                      sc_neg=data_curr_neg,
                                     ), code="""
            /* DEFINE VARIABLES
              var inds = slct_inds.value
            */
            {var_slct_str}


            {init_str}

            function checkMatch(i, source {param_str}) {{
               return {match_str};
            }}

            for (var i = 0; i <= source_pos.get_length(); i++){{
              if (checkMatch(i, source_pos {param_str})){{
                  {push_str_pos}
                  }}
              }}

            for (var i = 0; i <= source_neg.get_length(); i++){{
              if (checkMatch(i, source_neg {param_str})){{
                  {push_str_neg}
                  }}
              }}

            sc_pos.change.emit();
            sc_neg.change.emit();

        """.format(var_slct_str=var_slct_str, param_str=param_str, match_str= match_str,
                  push_str_neg=push_str_neg, push_str_pos=push_str_pos,
                  init_str=init_str,))

        selects = []
        for ind in ind_slct:

            list_slct = list(map(str, options_dict[ind]))
            slct = MultiSelect(size=1, value=[list_slct[0]], options=list_slct, title=ind)
            callback.args['slct_%s'%ind] = slct
            slct.js_on_change('value', callback)
            selects.append(slct)

        colors = brewer['Spectral'][len(cols_pos) + len(cols_neg) + 1]
        colors = [tuple(int(col.strip('#')[i:2+i], 16) for i in range(0,6,2)) + (0.9,)
                  for col in colors]

        list_p = []

        for valx, valy in xy_combs:

            make_str = lambda x, y: '%s = %s'%(x, y)
            title_str = ', '.join(make_str(ind_plt, val) for ind_plt, val
                                  in [(ind_pltx, valx), (ind_plty, valy)] if ind_plt)

            p = figure(plot_width=400, plot_height=300, title=title_str)

            for posneg, cols_list, data_curr in zip(['pos', 'neg'], [cols_pos, cols_neg],
                                                    [data_curr_pos, data_curr_neg]):
                color = {'pos': colors[:len(cols_pos)], 'neg': colors[len(cols_pos) + 1:len(cols_pos) + len(cols_neg) + 1]}

                areas = p.varea_stack(x="vre_scale", stackers=cols_list, color=color[posneg],
                                      legend=list(map(value, cols_list)),
                                      source=data_curr, view=views[(valx, valy)][posneg])

            p.legend.visible = False
            list_p.append(p)


        p_leg = figure(plot_height=100)
        areas_pos = p_leg.varea_stack(x="vre_scale", stackers=cols_pos, color=color['pos'],
                                      legend=list(map(value, cols_pos)), source=data_curr_pos)
        areas_neg = p_leg.varea_stack(x="vre_scale", stackers=cols_neg, color=color['neg'],
                                      legend=list(map(value, cols_neg)), source=data_curr_neg)
        areas = areas_pos + areas_neg


        return p