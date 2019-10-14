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
from bokeh.models.glyphs import MultiLine
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


    def _init_ind_lists(self):

        self.ind_plt = list(filter(lambda x: x is not None,
                                   [self.ind_pltx, self.ind_plty]))
        self.ind_slct = [x for x in self.ev.x_name
                         if not x in (self.ind_plt
                                      + [self.ind_axx]
                                      + self.cat_column)]

    @property
    def data(self):
        '''External modification of the data triggers. '''
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

        self._make_cols_lists()
        self.colors = self._get_color_list()
        self._make_index_lists()
        self._make_cds()
        self._make_views()
        self._init_callback()


    def _make_table(self):

        df = self._select_data()

        gpindex = self.ind_plt + self.ind_slct + [self.ind_axx]
        data = df.pivot_table(index=gpindex, columns=self.cat_column,
                              values=self.val_column)

        if len(self.cat_column) > 1:
            data.columns = [str(tuple(c)).replace('\'', '')
                            for c in data.columns]
        else:
            data.columns = [str(c) for c in data.columns]

        cols_to_str = {key: val for key, val in
                       {self.ind_pltx: lambda x: x[self.ind_pltx].apply(str),
                        self.ind_plty: lambda x: x[self.ind_plty].apply(str)}.items()
                       if key is not None}
        data = data.reset_index().assign(**cols_to_str).set_index(data.index.names)

        self.all_series = data.columns.tolist()

        if self.slct_series:
            data = data[self.slct_series]
            data = data.loc[~data.isna().all(axis=1)]

        self.data = data


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

        self.slct_list_dict = {ind: self.data.index.get_level_values(ind).unique().tolist()
                               for ind in self.data.index.names}

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

        slct_def = self.initial_selection

        self.cds_pos = (ColumnDataSource(self.data[self.cols_pos].xs(slct_def, level=self.ind_slct).reset_index())
                        if self.cols_pos else None)
        self.cds_neg = (ColumnDataSource(self.data[self.cols_neg].xs(slct_def, level=self.ind_slct).reset_index())
                        if self.cols_neg else None)
        self.cds_all_pos = (ColumnDataSource(self.data[self.cols_pos].reset_index())
                            if self.cols_pos else None)
        self.cds_all_neg = (ColumnDataSource(self.data[self.cols_neg].reset_index())
                            if self.cols_neg else None)


    def _make_views(self):

        get_flt = lambda ind, val: ([GroupFilter(column_name=ind, group=val)]
                                    if ind else [])

        def get_view(source, valx, valy):
            filters = (get_flt(self.ind_pltx, valx)
                       + get_flt(self.ind_plty, valy))
            return CDSView(source=source, filters=filters)

        list_pn = ['pos', 'neg']
        list_cds = [(cds, pn) for cds, pn in
                    zip([self.cds_pos, self.cds_neg], list_pn) if cds]

        self.views = dict.fromkeys(self.xy_combs)

        for valx, valy in self.views.keys():
            self.views[(valx, valy)] = dict.fromkeys(list_pn)

            for source, pn in list_cds:
                self.views[(valx, valy)][pn] = get_view(source, valx, valy)


    def _select_data(self):

        raise NotImplementedError(('%s must implement '
                                   '`_select_data`'%self.__class__))


    def get_js_args(self):

        get_datasource = lambda name: ({name: getattr(self, name)}
                                       if hasattr(self, name) else {})

        js_args = [get_datasource(name) for name
                   in ['cds_pos', 'cds_neg', 'cds_all_pos', 'cds_all_neg']]

        # list of dicts to single dict
        js_args = dict(itertools.chain.from_iterable(map(dict.items, js_args)))

        return js_args



    def _init_callback(self):

        cols_pos = self.cols_pos + self.ind_plt + [self.ind_axx, 'index']
        cols_neg = self.cols_neg + self.ind_plt + [self.ind_axx, 'index']
        js_string = JSCallbackCoder(self.ind_slct,
                                    cols_pos,
                                    cols_neg)()

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
        ncols = len(self.slct_list_dict[self.ind_pltx]) if self.ind_pltx else 1
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

class GeneralAreaPlot(GeneralPlot):


    def _make_single_plot(self, fig, data, view, cols, color):

        fig.varea_stack(x=self.ind_axx, stackers=cols, color=color,
                        legend=list(map(value, cols)),
                        source=data, view=view)

class GeneralBarPlot(GeneralPlot):


    def _make_single_plot(self, fig, data, view, cols, color):

        fig.vbar_stack(cols, x=self.ind_axx, width=0.9,
                       color=color, source=data,
                       legend=list(map(value, cols)))



if __name__ == '__main__':

    plot = BalancePlot(ev,
                        ind_axx='vre_scale',
                        ind_pltx='slot',
                        ind_plty=None,
                        cat_column=['func_no_slot'],
#                        slct_series=['phs_pchg', 'phs_pdch', 'batt_pchg', 'batt_pdch']
                        )

    self = plot

    show(plot._get_layout())

    print(JSCallbackCoder(self.ind_slct,
                    self.cols_pos,
                    self.cols_neg,
                    self.ind_plt + [self.ind_axx])())




