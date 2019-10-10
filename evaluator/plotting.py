#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:30:13 2019

@author: user
"""

import itertools
from bokeh.models.sources import ColumnDataSource, CustomJS
from bokeh.layouts import column, row, gridplot, WidgetBox
from bokeh.models import Legend, CDSView, GroupFilter
from bokeh.models.widgets import MultiSelect
from bokeh.plotting import figure
from bokeh.core.properties import value
from bokeh.palettes import brewer
from bokeh.io import show


class JSCallbackCoder():

    def __init__(self, ind_slct, cols_pos, cols_neg):

        self.ind_slct = ind_slct
        self.cols_pos = cols_pos
        self.cols_neg = cols_neg

        self._make_js_general_str()
        self._make_js_init_str()
        self._make_js_push_str()
        self._make_js_loop_str()
        self._make_js_emit_str()


    def _make_js_general_str(self):

        self.var_slct_str = '; '.join('var {inds} = slct_{inds}.value'.format(inds=inds)
                                      for inds in self.ind_slct) + ';'
        self.param_str = ', ' + ', '.join(self.ind_slct)
        self.match_str = ' && '.join('source.data[\'{inds}\'][i] == {inds}'.format(inds=inds)
                                     for inds in self.ind_slct)


    def _make_js_push_str(self):

        get_push_str = lambda pn: ''.join(('cds_{pn}.data[\'{col}\']'
                                           '.push(cds_all_{pn}.'
                                           'data[\'{col}\'][i]); '
                                           ).format(pn=pn, col=col)
                                for col in getattr(self, 'cols_%s'%pn))

        self.push_str_pos = get_push_str('pos')
        self.push_str_neg = get_push_str('neg')


    def _make_js_init_str(self):

        self.init_str = ''.join('cds_%s.data[\'%s\'] = []; '%(pn, col)
                                for pn in ['pos', 'neg']
                                for col in getattr(self, 'cols_%s'%pn))


    def _make_js_loop_str(self):

        get_loop_str = lambda pn: '''
            for (var i = 0; i <= cds_all_{pn}.get_length(); i++){{
                if (checkMatch(i, cds_all_{pn}{param_str})){{
                {push_str}
              }}
            }}'''.format(pn=pn, param_str=self.param_str,
                         push_str=getattr(self, 'push_str_%s'%pn)
                         ) if getattr(self, 'cols_%s'%pn) else ''

        self.loop_str_pos = get_loop_str('pos')
        self.loop_str_neg = get_loop_str('neg')


    def _make_js_emit_str(self):

        get_emit_str = lambda pn: ''' cds_%s.change.emit();
                            '''%pn if getattr(self, 'cols_%s'%pn) else ''

        self.emit_str = get_emit_str('pos') + get_emit_str('neg')


    def get_js_string(self):

        # init JS callback object --> parent
        js_code = """
            {var_slct_str}

            {init_str}
            function checkMatch(i, source{param_str}) {{
               return {match_str};
            }}
            {loop_str_pos}
            {loop_str_neg}

            {emit_str}
            """.format(var_slct_str=self.var_slct_str,
                       param_str=self.param_str,
                       match_str=self.match_str,
                       init_str=self.init_str,
                       loop_str_neg=self.loop_str_neg,
                       loop_str_pos=self.loop_str_pos,
                       emit_str=self.emit_str)

        return js_code


    def __call__(self):

        return self.get_js_string()

    def __repr__(self):

        return self.get_js_string()

class SymenergyPlotter():
    '''
    Base class
    '''

    cat_column = None  # defined by children
    val_column = None  # defined by children
    cols_neg = []

        print('Init SymenergyPlotter')
    def __init__(self, ev, ind_axx, ind_pltx, ind_plty, slct_series=None,
                 cat_column=None):

        self.ev = ev
        self.ind_pltx = ind_pltx
        self.ind_plty = ind_plty
        self.ind_axx = ind_axx

        if cat_column:
            self.cat_column = cat_column

        self.slct_series = slct_series

        self._init_ind_lists()
        self._make_table()

        self.colors = self._get_color_list()

    def _init_ind_lists(self):
        # init indices
        self.ind_plt = list(filter(lambda x: x is not None,
                                   [self.ind_pltx,
                                    self.ind_plty]))
        self.ind_slct = [x for x in self.ev.x_name
                         if not x in (self.ind_plt
                                      + [self.ind_axx]
                                      + self.cat_column)]

        self._make_index_lists()
        self._make_cds()
        self._make_views()
        self._init_callback()

    def _select_data(self):
        '''
        Implemented in children. Select DataFrame from Evaluator attributes.
        '''

    def _make_table(self):

        df = self._select_data()

        dfgp = df.pivot_table(index=self.ind_slct + self.ind_all,
                              columns=self.cat_column,  values=self.val_column)
        dfgp.columns.names = [None]

        dfgp = dfgp[self.slct_series]
        if len(self.cat_column) > 1:
            data.columns = [str(tuple(c)).replace('\'', '')
                            for c in data.columns]
        else:
            data.columns = [str(c) for c in data.columns]

        cols_to_str = {key: val for key, val in
                       {self.ind_pltx: lambda x: x[self.ind_pltx].apply(str),
                        self.ind_plty: lambda x: x[self.ind_plty].apply(str)}.items()
                       if key is not None}
        dfgp = dfgp.reset_index().assign(**cols_to_str).set_index(dfgp.index.names)

        self.dfgp = dfgp


    def _make_cols_lists(self):

        if len(self.cat_column) > 1:
            self.cols_neg = [c for c in self.data.columns
                             if any(sub_c.endswith(pat)
                                    for pat in self.cols_neg for sub_c in c)]
        else:
            self.cols_neg = [c for c in self.data.columns
                             if any(c.endswith(pat) for pat in self.cols_neg)]
        self.cols_pos = [c for c in self.data.columns if c not in self.cols_neg]


    def _get_xy_list(self, ind):

        return self.slct_list_dict[ind] if ind else [None]


    def _make_index_lists(self):

        self.slct_list_dict = {ind: self.dfgp.index.get_level_values(ind).unique().tolist()
                               for ind in self.dfgp.index.names}

        self.xy_combs = list(itertools.product(self._get_xy_list(self.ind_pltx),
                                               self._get_xy_list(self.ind_plty)
                                               ))
    @property
    def initial_selection(self):
        '''
        Initial data selection (MultiSelect widgets) defaults to first data row
        '''

        return getattr(self, '_initial_selection',
                       tuple(self.data.reset_index()[self.ind_slct].iloc[0]))

    @initial_selection.setter
    def initial_selection(self, val):

        self._initial_selection = val
        self._make_cds()

    def _make_cds(self):

        # initial selection

        self.cds_pos = (ColumnDataSource(self.dfgp[self.cols_pos].loc[slct_def].reset_index())
        slct_def = self.initial_selection
                        if self.cols_pos else None)
        self.cds_neg = (ColumnDataSource(self.dfgp[self.cols_neg].loc[slct_def].reset_index())
                        if self.cols_neg else None)
        self.cds_all_pos = (ColumnDataSource(self.dfgp[self.cols_pos].reset_index())
                            if self.cols_pos else None)
        self.cds_all_neg = (ColumnDataSource(self.dfgp[self.cols_neg].reset_index())
                            if self.cols_neg else None)

    def _make_views(self):

        def get_view(source, valx, valy):

            get_flt = lambda ii, vv: ([GroupFilter(column_name=ii, group=vv)]
                                      if ii else [])

            filters = get_flt(self.ind_pltx, valx) + get_flt(self.ind_plty, valy)
            return CDSView(source=source, filters=filters)

        list_cds = [(cds, pn) for cds, pn in
                    zip([self.cds_pos, self.cds_neg], ['pos', 'neg']) if cds]

        self.views = {(valx, valy): {posneg: get_view(source, valx, valy)
                                for source, posneg in list_cds}
                 for valx, valy in self.xy_combs}








    def get_js_args(self):

        get_datasource = lambda name: ({name: getattr(self, name)}
                                       if hasattr(self, name) else {})

        js_args = [get_datasource(name) for name
                   in ['cds_pos', 'cds_neg', 'cds_all_pos', 'cds_all_neg']]

        # list of dicts to single dict
        js_args = dict(itertools.chain.from_iterable(map(dict.items, js_args)))

        return js_args



    def _init_callback(self):
        js_string = JSCallbackCoder(self.ind_slct,
                                    self.cols_pos,
                                    self.cols_neg)()

        self.callback = CustomJS(args=self.get_js_args(),
                                 code=js_string)

    def _get_multiselects(self):

        selects = []
        for nind, ind in enumerate(self.ind_slct):

            list_slct = list(map(str, self.slct_list_dict[ind]))
            slct = MultiSelect(size=1,
                               value=[str(self.initial_selection[nind])],
                               options=list_slct, title=ind)
            self.callback.args['slct_%s'%ind] = slct
            slct.js_on_change('value', self.callback)
            selects.append(slct)

        return selects

    def _get_color_list(self):

        # init color list --> parent if cols_posneg empty
        ncolors = len(self.cols_pos) + len(self.cols_neg) + 1

        colorset = 'Set3'
        maxcolors = max(brewer[colorset].keys())
        mincolors = min(brewer[colorset].keys())
        if ncolors <= maxcolors:
            colors = brewer[colorset][max(mincolors, ncolors)][:ncolors]
        else:
            colorcycler = itertools.cycle(brewer[colorset][maxcolors])
            colors = list(zip(*zip(range(ncolors), colorcycler)))[1]

        # convert to rgba --> parent
        colors = [tuple(int(col.strip('#')[i:2+i], 16) for i in range(0,6,2)) + (0.9,)
                  for col in colors]

        colors_posneg = {
                'pos': colors[:len(self.cols_pos)],
                'neg': colors[len(self.cols_pos) + 1:len(self.cols_pos)
                                + len(self.cols_neg) + 1]}

        return colors_posneg

    def _get_plot_list(self):

        list_p = []

        for valy in self._get_xy_list(self.ind_plty):
            for valx in self._get_xy_list(self.ind_pltx):

                make_str = lambda x, y: '%s = %s'%(x, y)
                title_str = ', '.join(make_str(ind_plt, val)
                                      for ind_plt, val
                                      in [(self.ind_pltx, valx),
                                          (self.ind_plty, valy)]
                                      if ind_plt)

                p = figure(plot_width=400, plot_height=300, title=title_str)

                posneg_vars = zip(['pos', 'neg'],
                                  [self.cols_pos, self.cols_neg],
                                  [self.cds_pos, self.cds_neg])
                for posneg, cols, data in posneg_vars:

                    view = self.views[(valx, valy)][posneg]

                    self._make_single_plot(fig=p, color=self.colors[posneg],
                                           data=data, view=view, cols=cols)

                p.legend.visible = False
                list_p.append(p)

        return list_p

    def _make_layout(self):

        ''''''

    def _get_layout(self):

        list_p = self._get_plot_list()
        selects = self._get_multiselects()
        ncols = len(self.slct_list_dict[self.ind_pltx])
        p_leg = self._get_legend()

        controls = WidgetBox(*selects)
        layout = column(row(controls,
                            p_leg
                            ), gridplot(list_p, ncols=ncols))

        return layout

    def __call__(self):

        return self._get_layout()

    def _get_legend(self):
        '''
        Return empty plot with shared legend.
        '''

        p_leg = figure(plot_height=100)

        get_leg_areas = lambda cols, pn, cds: \
            p_leg.varea_stack(x=self.ind_axx, stackers=cols,
                              color=self.colors[pn],
                              legend=list(map(value, cols)), source=cds)

        areas_pos = get_leg_areas(self.cols_pos, 'pos', self.cds_pos) if hasattr(self, 'cds_pos') else []
        areas_neg = get_leg_areas(self.cols_neg, 'neg', self.cds_neg) if hasattr(self, 'cds_neg') else []
        areas = areas_pos + areas_neg

        for rend in p_leg.renderers:
            rend.visible = False

        p_leg.xaxis.visible = False
        p_leg.yaxis.visible = False
        p_leg.xgrid.visible = False
        p_leg.ygrid.visible = False
        p_leg.outline_line_color = None
        p_leg.toolbar.logo = None
        p_leg.toolbar_location = None
        p_leg.legend.visible = False
        p_leg.background_fill_alpha = 0
        p_leg.border_fill_alpha = 0

        legend_handles_labels = list(zip(self.cols_pos + self.cols_neg,
                                         map(lambda x: [x], areas)))
        legend = Legend(items=legend_handles_labels, location=(0.5, 0))
        legend.click_policy="mute"
        legend.orientation = "horizontal"
        p_leg.add_layout(legend, 'above')

        return p_leg


class BalancePlot(SymenergyPlotter):

    cat_column = ['func_no_slot']
    val_column = 'lambd'
    cols_neg = ['l', 'pchg', 'curt_p']


    def _select_data(self):

        df = self.ev.df_bal
        df = df.loc[-df.func_no_slot.str.contains('tc', 'lam')
                   & -df.slot.isin(['global'])
                   & -df.pwrerg.isin(['erg'])]

        df.loc[:, 'lambd'] = df.lambd.astype(float)
        df = df.sort_values(self.ind_axx)

        return df


    def _make_single_plot(self, fig, data, view, cols, color):

        fig.varea_stack(x=self.ind_axx, stackers=cols, color=color,
                        legend=list(map(value, cols)),
                        source=data, view=view)

class GeneralPlot(SymenergyPlotter):

    val_column = 'lambd'
    cols_neg = ['l', 'pchg', 'curt_p']

    def _select_data(self):

        df = self.ev.df_exp.loc[self.ev.df_exp.is_optimum]

        df.loc[:, 'lambd'] = df.lambd.astype(float)
        df = df.sort_values(self.ind_axx)

        return df


    def _make_single_plot(self, fig, data, view, cols, color):

        for column_slct, color_slct in zip(cols, color):

            fig.circle(x=self.ind_axx, y=column_slct, color=color_slct,
                       source=data, view=view, line_color='DimGrey')
            fig.line(x=self.ind_axx, y=column_slct, color=color_slct,
                     source=data, view=view)


if __name__ == '__main__':

    balplot = BalancePlot(ev, ind_axx='vre_scale',
                          ind_pltx='eff_phs',
                          ind_plty='slot')


    show(balplot._get_layout())




# %%

#
#class SymenergyPlotting():
#    '''
#    Mixin class for Evaluator.
#    '''
#
#    def make_supply_shadow_price_plot(self, ind_pltx='slot',
#                                      ind_plty=None, ind_axx='vre_scale'):
#
#        df = self.df_exp.query('func_no_slot == "pi_load"')
#
#
#
#        ind_pltx='slot'
#        ind_plty=None
#        ind_axx='vre_scale'
#
#
#
#
#    def make_energy_balance_plot(self, ind_pltx='slot',
#                                 ind_plty=None, ind_axx='vre_scale'):
#
#
## %%
#
#        # select data
#        df = self.df_bal
#        df = df.loc[-df.func_no_slot.str.contains('tc', 'lam')
#                   & -df.slot.isin(['global'])
#                   & -df.pwrerg.isin(['erg'])]
#
#        df['lambd'] = df.lambd.astype(float)
#        df = df.sort_values(ind_axx)
#
#
#        # init indices
#        ind_all = list(filter(lambda x: x is not None, [ind_axx, ind_pltx, ind_plty]))
#        ind_slct = [x for x in self.x_name if not x in ind_all]
#
#
#        # make datatable --> children, since selects value column
#        dfgp = df.pivot_table(index=ind_slct + ind_all,
#                              columns='func_no_slot',  values='lambd')
#        dfgp.columns.names = [None]
#
#        # cast index values as strings --> parent class
#        cols_to_str = {key: val for key, val in
#                       {ind_pltx: lambda x: x[ind_pltx].apply(str),
#                        ind_plty: lambda x: x[ind_plty].apply(str)}.items()
#                       if key is not None}
#        dfgp = dfgp.reset_index().assign(**cols_to_str).set_index(dfgp.index.names)
#
#        # all pos/neg logic in balance class --> child
#        cols_neg = ['l', 'pchg', 'curt_p']
#        cols_neg = [c for c in dfgp.columns
#                    if any(c.endswith(pat) for pat in cols_neg)]
#        cols_pos = [c for c in dfgp.columns if c not in cols_neg]
#
#        # list of all indices + combinations thereof --> parent
#        slct_list_dict = {ind: dfgp.index.get_level_values(ind).unique().tolist()
#                          for ind in dfgp.index.names}
#        xy_combs = list(itertools.product(slct_list_dict[ind_pltx] if ind_pltx else [None],
#                                          slct_list_dict[ind_plty] if ind_plty else [None]))
#
#        # initial selection
#        slct_def = list(dfgp.iloc[[0]].index)[0][:len(ind_slct)]
#
#        # definition of datasources --> all pos/neg logic in balance class --> child
#        if cols_pos:
#            self.cds_pos=ColumnDataSource(dfgp[cols_pos].loc[slct_def].reset_index())
#        if cols_neg:
#            self.cds_neg=ColumnDataSource(dfgp[cols_neg].loc[slct_def].reset_index())
#        if cols_pos:
#            self.cds_all_pos=ColumnDataSource(dfgp[cols_pos].reset_index())
#        if cols_neg:
#            self.cds_all_neg=ColumnDataSource(dfgp[cols_neg].reset_index())
#
#        # view definition function and index combinations of --> parent method
#        def get_view(source, valx, valy):
#
#            get_flt = lambda ii, vv: ([GroupFilter(column_name=ii, group=vv)]
#                                      if ii else [])
#
#            filters = get_flt(ind_pltx, valx) + get_flt(ind_plty, valy)
#            return CDSView(source=source, filters=filters)
#
#
#        # define data views --> all pos/neg logic in balance class --> child
#        views = {(valx, valy): {posneg: get_view(source, valx, valy)
#                                for source, posneg
#                                in zip([self.cds_pos, self.cds_neg], ['pos', 'neg'])}
#                 for valx, valy in xy_combs}
#
#
#        # js string for selection definition --> parent
#        var_slct_str = '; '.join('var {inds} = slct_{inds}.value'.format(inds=inds)
#                for inds in ind_slct) + ';'
#        param_str = ', ' + ', '.join(ind_slct)
#        match_str = ' && '.join('source.data[\'{inds}\'][i] == {inds}'.format(inds=inds) for inds in ind_slct)
#
#        # affects series names --> children
#        def get_push_str(posneg, plant, var):
#            return ('cds_{pn}.data[\'{plant}{var}\']'
#                    '.push(cds_all_{pn}.data[\'{plant}{var}\'][i]); '
#                   ).format(pn=posneg, plant=plant, var=var)
#
#        push_str_pos = ''.join([
#            ''.join([get_push_str('pos', store_name, '_pdch') for store_name in self.model.storages]),
#            ''.join([get_push_str('pos', plant_name, '_p') for plant_name in self.model.plants]),
#            get_push_str('pos', 'vre', '')
#        ])
#
#        push_str_neg = ''.join([
#            get_push_str('neg', 'l', ''),
#            get_push_str('neg', 'curt', '_p') if 'curt' in self.model.comps else '',
#            ''.join([get_push_str('neg', store_name, '_pchg') for store_name in self.model.storages]),
#        ])
#
#        # initialize series lists --> children
#        def get_init_str(list_p, list_var, list_pn):
#            return ''.join('cds_%s.data[\'%s%s\'] = []; '%(pn, p, var)
#                for p, var, pn
#                in [(p, *vpn)for p, vpn in itertools.product(list_p, zip(list_var, list_pn))]
#            )
#
#        init_str = '; '.join([
#            get_init_str(['l'], [''], ['neg']),
#            get_init_str(['vre'], [''], ['pos']),
#            get_init_str(self.model.storages, ['_pdch', '_pchg'], ['pos', 'neg']),
#            get_init_str(self.model.plants, ['_p'], ['pos']),
#            get_init_str(['curt'], ['_p'], ['neg']) if 'curt' in self.model.comps else '',
#        ])
#
#        # as long as only existing cds_all_ are set --> parent
#        get_loop_str = lambda source_str, push_str: '''
#            for (var i = 0; i <= {source_str}.get_length(); i++){{
#              if (checkMatch(i, {source_str}{param_str})){{
#                {push_str}
#              }}
#            }}'''.format(push_str=push_str, param_str=param_str,
#                         source_str=source_str) if hasattr(self, source_str) else ''
#
#        loop_str_pos = get_loop_str('cds_all_pos', push_str_pos)
#        loop_str_neg = get_loop_str('cds_all_neg', push_str_neg)
#
#
#        # pass all ColumnDataSources to CustomJS --> parent
#        get_datasource = lambda name: ({name: getattr(self, name)}
#                                       if hasattr(self, name) else {})
#        js_args = [get_datasource(name) for name
#                   in ['cds_pos', 'cds_neg', 'cds_all_pos', 'cds_all_neg']]
#        js_args = dict(itertools.chain.from_iterable(map(dict.items, js_args)))
#
#
#
#
#        get_emit_str = lambda cds: ''' {cds}.change.emit();
#                            '''.format(cds=cds) if hasattr(self, cds) else ''
#
#        emit_str = get_emit_str('cds_pos') + get_emit_str('cds_neg')
#
#        # init JS callback object --> parent
#        callback = CustomJS(args=js_args, code="""
#            {var_slct_str}
#            {init_str}
#            function checkMatch(i, source{param_str}) {{
#               return {match_str};
#            }}
#            {loop_str_pos}
#            {loop_str_neg}
#            {emit_str}
#            """.format(var_slct_str=var_slct_str, param_str=param_str,
#                       match_str=match_str, init_str=init_str,
#                       loop_str_neg=loop_str_neg, loop_str_pos=loop_str_pos,
#                       emit_str=emit_str))
#
#        selects = []
#        for ind in ind_slct:
#
#            list_slct = list(map(str, slct_list_dict[ind]))
#            slct = MultiSelect(size=1, value=[list_slct[0]],
#                               options=list_slct, title=ind)
#            callback.args['slct_%s'%ind] = slct
#            slct.js_on_change('value', callback)
#            selects.append(slct)
#
#        # init color list --> parent if cols_posneg empty
#        colors = brewer['Spectral'][len(cols_pos) + len(cols_neg) + 1]
#
#        # convert to rgba --> parent
#        colors = [tuple(int(col.strip('#')[i:2+i], 16) for i in range(0,6,2)) + (0.9,)
#                  for col in colors]
#
#        list_p = []
#
#        for valx, valy in xy_combs:
#
#            make_str = lambda x, y: '%s = %s'%(x, y)
#            title_str = ', '.join(make_str(ind_plt, val) for ind_plt, val
#                                  in [(ind_pltx, valx), (ind_plty, valy)]
#                                  if ind_plt)
#
#            p = figure(plot_width=400, plot_height=300, title=title_str)
#
#            posneg_vars = zip(['pos', 'neg'], [cols_pos, cols_neg],
#                              [self.cds_pos, self.cds_neg])
#            for posneg, cols_list, data_curr in posneg_vars:
#                color = {'pos': colors[:len(cols_pos)],
#                         'neg': colors[len(cols_pos) + 1:len(cols_pos) + len(cols_neg) + 1]}
#
#                areas = p.varea_stack(x=ind_axx, stackers=cols_list, color=color[posneg],
#                                      legend=list(map(value, cols_list)),
#                                      source=data_curr, view=views[(valx, valy)][posneg])
#
#            p.legend.visible = False
#            list_p.append(p)
#
#
## =============================================================================
## =============================================================================
## #         p_leg = figure(plot_height=100)
## #
## #         get_leg_areas = lambda cols, pn, cds: \
## #             p_leg.varea_stack(x=ind_axx, stackers=cols,
## #                               color=color[pn],
## #                               legend=list(map(value, cols)),
## #                               source=cds)
## #
## #         areas_pos = get_leg_areas(cols_pos, 'pos', self.cds_pos) if hasattr(self, 'cds_pos') else []
## #         areas_neg = get_leg_areas(cols_neg, 'neg', self.cds_neg) if hasattr(self, 'cds_neg') else []
## #         areas = areas_pos + areas_neg
## #
## #         for rend in p_leg.renderers:
## #             rend.visible = False
## #         p_leg.xaxis.visible = False
## #         p_leg.yaxis.visible = False
## #         p_leg.xgrid.visible = False
## #         p_leg.ygrid.visible = False
## #         p_leg.outline_line_color = None
## #         p_leg.toolbar.logo = None
## #         p_leg.toolbar_location = None
## #         p_leg.legend.visible = False
## #         p_leg.background_fill_alpha = 0
## #         p_leg.border_fill_alpha = 0
## # #
## #         legend_handles_labels = list(zip(cols_pos + cols_neg, map(lambda x: [x], areas)))
## #         legend = Legend(items=legend_handles_labels, location=(0.5, 0))
## #         legend.click_policy="mute"
## #         legend.orientation = "horizontal"
## #         p_leg.add_layout(legend, 'above')
## =============================================================================
## =============================================================================
#
#        controls = WidgetBox(*selects)
#        layout = column(row(controls,
#                            p_leg
#                            ), gridplot(list_p, ncols=len(slct_list_dict[ind_pltx])))
#
#        show(layout)
#
#        #return layout