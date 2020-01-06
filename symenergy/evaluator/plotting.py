#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
The plotting module contains various classes to generate typical SymEnergy
output plots. Data update callbacks are implemented in JavaScript. This allows
for interactive standalone plots, suited for Jupyter Notebooks.

Plots are arranged in 2D arrays. The two dimensions may correspond to any
of the parameters specified by the
:class:`symenergy.evaluator.evaluator.Evaluator`'s `df_exp` table.
Plot objects are typically

.. code-block:: python

    CustomPlot(Ev(df=df, x_name=category_cols),
                  ind_axx='catx',
                  ind_pltx='cat2',
                  ind_plty=None,
                  cat_column=['cat1'])

All classes return Bokeh Layout objects. They can be displayed as

- standalone html documents using `bokeh.io.output_file`
- interactive plots in Jupyter notebooks using `bokeh.io.output_notebooks`




Subclassing SymenergyPlotter
----------------------------

The :class:`symenergy.evaluator.plotting.SymenergyPlotter` class acts as a
base class for the more targeted plots documented below. It takes care of
most Bokeh logic, including the generation of JavaScript callbacks. Similar to
the :class:`symenergy.evaluator.plotting.BalancePlot` class, it can easily be
subclassed to generate customized plots.

For this purpose, a class with two methods is defined:

* the method `_select_data(self)` to obtain (filtered) data from the `ev`
  ("Evaluator"). It must return a DataFrame with value and index columns.
* the method `_make_single_plot(self, fig, data, view, cols, color)` adds
  bokeh plots to the figure `fig` provided as an argument. `data` is the
  `ColumnDataSource` and is simply passed as `source` argument to the Bokeh
  plot. The same holds for the `view` parameter, which corresponds to the
  Bokeh plot `view` argument. `cols` and `color` are the data series and
  the corresponding colors, respectively.

The `SymenergyPlotter` is initialized with a
:class:`symenergy.evaluator.evaluator.Evaluator` object. However, only the
`df` (dataframe) and `x_name` (category column names) attributes are used by the plotting
class. Because of this, a mock evaluator object can also serve as input,
containing only these two attributes. Consequently, this plotting module can
be used with any kind of input data, not necessarily originating from the
`SymEnergy` model. This is shown in the example below.

.. code-block:: python

    from collections import namedtuple
    import numpy as np

    # generate input data
    category_cols = ['cat1', 'cat2', 'cat3', 'catx']
    cat1 = ['A', 'B', 'C']
    cat2 = ['A1', 'A2']
    cat3 = ['X', 'Y', 'Z']
    catx = range(20)
    df = pd.DataFrame(itertools.product(cat1, cat2, cat3, catx),
                      columns=category_cols)
    df['value'] = np.random.random(len(df)) * 20

    Ev = namedtuple('Ev', ['df', 'x_name'])
    ev = Ev(df=df, x_name=category_cols)

    #
    class CustomPlot(SymenergyPlotter):

        val_column = 'value'
        cols_neg = []

        def _select_data(self):

            df = self.ev.df
            return df


        def _make_single_plot(self, fig, data, view, cols, color):

            for column_slct, color_slct in zip(cols, color):

                fig.circle(x=self.ind_axx, y=column_slct, color=color_slct,
                           source=data, view=view, line_color='DimGrey')
                fig.line(x=self.ind_axx, y=column_slct, color=color_slct,
                         source=data, view=view)



    plot = CustomPlot(Ev(df=df, x_name=category_cols),
                      ind_axx='catx',
                      ind_pltx='cat2',
                      ind_plty=None,
                      cat_column=['cat1'])

    show(plot._get_layout())

```

To generate stacked barplots the `_make_single_plot` method would be written
as follows.

```python
    def _make_single_plot(self, fig, data, view, cols, color):

        fig.vbar_stack(cols, x=self.ind_axx, width=0.9,
                       color=color, source=data,
                       legend=list(map(value, cols)),
                       view=view)
```

"""

import itertools
import pandas as pd
from bokeh.models.sources import ColumnDataSource, CustomJS
from bokeh.layouts import column, row, gridplot, WidgetBox
from bokeh.models import Legend, CDSView, GroupFilter
from bokeh.models.widgets import MultiSelect
from bokeh.plotting import figure
from bokeh.models.glyphs import MultiLine
from bokeh.core.properties import value
from bokeh.palettes import brewer
from bokeh.io import show

pd.set_option('mode.chained_assignment', None)

class JSCallbackCoder():
    '''
    Parameters
    ----------
    ind_slct : list(str)
        names of the indices selected through the MultiSelect widgets
    cols_series : list(str)
        names of all data series
    cols_pos : list(str)
        names of positive datasource columns (including indices)
    cols_neg : list(str)
        names of negative datasource columns (including indices)
    '''


    def __init__(self, ind_slct, cols_series, cols_pos, cols_neg):

        self.ind_slct = ind_slct
        self.cols_pos = cols_pos
        self.cols_neg = cols_neg
        self.cols_series = cols_series

        self._make_js_general_str()
        self._make_js_init_str()
        self._make_js_push_str()
        self._make_js_loop_str()
        self._make_js_emit_str()


    def _make_js_general_str(self):

        self.var_slct_str = '; '.join('var {inds} = slct_{inds}.value'.format(inds=inds)
                                      for inds in self.ind_slct) + ';'
        self.series_slct_str = 'var dataseries = slct_dataseries.value' + ';'
        self.param_str = ', ' + ', '.join(self.ind_slct)
        self.match_str = ' && '.join('source.data[\'{inds}\'][i] == {inds}'.format(inds=inds)
                                     for inds in self.ind_slct)


    def _make_js_push_str(self):

        def get_push_str(pn):
            for col in getattr(self, 'cols_%s'%pn):
                tr_block = 'cds_all_{pn}.data[\'{col}\'][i]'
                if col in self.cols_series:
                    if_block = 'dataseries.includes(\'{col}\')'
                    fa_block = '0'
                    val = '({}) ? ({}) : ({})'.format(if_block, tr_block, fa_block)
                else:
                    val = tr_block
                yield ('cds_{pn}.data[\'{col}\'].push(%s); \n'%val).format(pn=pn, col=col)

        self.push_str_pos = ''.join(get_push_str('pos'))
        self.push_str_neg = ''.join(get_push_str('neg'))


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
            {series_slct_str}

            {init_str}
            function checkMatch(i, source{param_str}) {{
               return {match_str};
            }}
            {loop_str_pos}
            {loop_str_neg}

            {emit_str}
            """.format(var_slct_str=self.var_slct_str,
                       series_slct_str=self.series_slct_str,
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
    Base class for the
    '''

    cat_column = None  # defined by children
    val_column = None  # defined by children
    cols_neg = []

    def __init__(self, ev, ind_axx, ind_pltx, ind_plty, slct_series=None,
                 cat_column=None, plot_width=400, plot_height=300,
                 ind_axy=None):

        self.ev = ev
        self.ind_pltx = ind_pltx
        self.ind_plty = ind_plty
        self.ind_axx = ind_axx
        self.ind_axy = ind_axy

        self.plot_height = plot_height
        self.plot_width = plot_width

        if cat_column:
            self.cat_column = cat_column

        self.slct_series = slct_series

        self._init_ind_lists()
        self._make_table()


    def _init_ind_lists(self):

        self.ind_plt = list(filter(lambda x: x is not None,
                                   [self.ind_pltx, self.ind_plty]))
        self.ind_slct = [x for x in self.ev.x_name
                         if not x in (self.ind_plt + [self.ind_axx]
                                      + ([self.ind_axy] if self.ind_axy else [])
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

        gpindex = (self.ind_plt + self.ind_slct + [self.ind_axx]
                   + ([self.ind_axy] if self.ind_axy else []))
        data = df.pivot_table(index=gpindex, columns=self.cat_column,
                              values=self.val_column, fill_value=0)

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

            missing = [c for c in self.slct_series if not c in self.all_series]
            if missing:
                raise IndexError(('Unknown data series %s; options are %s.'
                                  )%(str(missing), str(self.all_series)))

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

        df_all_pos = self.data[self.cols_pos].reset_index()
        self.cds_all_pos = (ColumnDataSource(df_all_pos)
                            if self.cols_pos else None)
        df_all_neg = self.data[self.cols_neg].reset_index()
        self.cds_all_neg = (ColumnDataSource(df_all_neg)
                            if self.cols_neg else None)
        if slct_def:
            df_pos = self.data[self.cols_pos].xs(slct_def, level=self.ind_slct)
            self.cds_pos = (ColumnDataSource(df_pos.reset_index())
                            if self.cols_pos else None)
            df_neg = self.data[self.cols_neg].xs(slct_def, level=self.ind_slct)
            self.cds_neg = (ColumnDataSource(df_neg.reset_index())
                            if self.cols_neg else None)
        else:
            self.cds_pos = self.cds_all_pos
            self.cds_neg = self.cds_all_neg


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
                                       if hasattr(self, name)
                                       and getattr(self, name) else {})

        js_args = [get_datasource(name) for name
                   in ['cds_pos', 'cds_neg', 'cds_all_pos', 'cds_all_neg']]

        # list of dicts to single dict
        js_args = dict(itertools.chain.from_iterable(map(dict.items, js_args)))

        return js_args



    def _init_callback(self):

        cols_ind = (self.ind_plt + [self.ind_axx, 'index']
                    + ([self.ind_axy] if self.ind_axy else []))
        cols_pos = self.cols_pos + (cols_ind if self.cols_pos else [])
        cols_neg = self.cols_neg + (cols_ind if self.cols_neg else [])
        cols_series = self.cols_pos + self.cols_neg
        js_string = JSCallbackCoder(self.ind_slct,
                                    cols_series,
                                    cols_pos,
                                    cols_neg)()

        self.callback = CustomJS(args=self.get_js_args(),
                                 code=js_string)


    def _get_series_multiselect(self):

        list_slct = self.cols_pos + self.cols_neg
        slct = MultiSelect(size=len(list_slct),
                           value=list_slct, options=list_slct)

        self.callback.args['slct_dataseries'] = slct
        slct.js_on_change('value', self.callback)

        return slct


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
        colors = [tuple(int(col.strip('#')[i:2+i], 16)
                            for i in range(0,6,2)) + (0.9,)
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

                p = figure(plot_width=self.plot_width,
                           plot_height=self.plot_height,
                           title=title_str,
                           x_axis_label=self.ind_axx,
                           y_axis_label=(self.ind_axy if self.ind_axy
                                         else self.val_column))

                posneg_vars = zip(['pos', 'neg'],
                                  [self.cols_pos, self.cols_neg],
                                  [self.cds_pos, self.cds_neg])
                posneg_vars = [vars_ for vars_ in posneg_vars
                               if vars_[1]]  # skip if columns empty (e.g. no neg)
                for posneg, cols, data in posneg_vars:

                    view = self.views[(valx, valy)][posneg]

                    self._make_single_plot(fig=p, color=self.colors[posneg],
                                           data=data, view=view, cols=cols)

                if p.legend:
                    p.legend.visible = False
                list_p.append(p)

        return list_p


    def _get_layout(self):

        list_p = self._get_plot_list()
        selects = self._get_multiselects()
        select_ds = self._get_series_multiselect()
        ncols = len(self.slct_list_dict[self.ind_pltx]) if self.ind_pltx else 1
        p_leg = self._get_legend()

        controls = WidgetBox(*selects)
        layout = column(select_ds, row(controls,
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

        areas_pos = (get_leg_areas(self.cols_pos, 'pos', self.cds_pos)
                                   if hasattr(self, 'cds_pos') else [])
        areas_neg = (get_leg_areas(self.cols_neg, 'neg', self.cds_neg)
                                   if hasattr(self, 'cds_neg') else [])
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
    '''
    Generates power balance plots from the data in a
    :class:`symenergy.evaluator.Evaluator` object.

    - only the optimum solution is shown for each set of parameters
    - demand-like power (demand, charging, curtailment) is shown negative
    - the energy balance sums up to zero
    - stacked area plots are used

    Continuing the example from
    :func:`symenergy.evaluator.evaluator.Evaluator.expand_to_x_vals_parallel`:

    .. code-block:: python
        >>> from bokeh.io import show
        >>> from bokeh.plotting import output_notebook, output_file
        >>> import symenergy.evaluator.plotting as plotting

        >>> output_notebook()
        >>> output_file('example.html')
        >>> balplot = plotting.BalancePlot(ev=ev, ind_axx='vre_scale_none',
                                           ind_pltx='slot', ind_plty=None)
        >>> show(balplot._get_layout())

    ---------------------------------------------------------------------------

    .. image:: _static/example_balanceplot.png
        :align: center
        :alt: figure example balanceplot

    ---------------------------------------------------------------------------

    '''

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
                       legend=list(map(value, cols)),
                       view=view)



if __name__ == '__main__':

    #

    from collections import namedtuple
    import numpy as np

    category_cols = ['cat1', 'cat2', 'cat3', 'catx']
    cat1 = ['A', 'B', 'C']
    cat2 = ['A1', 'A2']
    cat3 = ['X', 'Y', 'Z']
    catx = range(20)
    df = pd.DataFrame(itertools.product(cat1, cat2, cat3, catx),
                      columns=category_cols)
    df['value'] = np.random.random(len(df)) * 20


    Ev = namedtuple('Ev', ['df', 'x_name'])
    ev = Ev(df=df, x_name=category_cols)


    class CustomPlot(SymenergyPlotter):

        val_column = 'value'
        cols_neg = []

        def _select_data(self):

            df = self.ev.df
            return df


        def _make_single_plot(self, fig, data, view, cols, color):

            for column_slct, color_slct in zip(cols, color):

                fig.circle(x=self.ind_axx, y=column_slct, color=color_slct,
                           source=data, view=view, line_color='DimGrey')
                fig.line(x=self.ind_axx, y=column_slct, color=color_slct,
                         source=data, view=view)



    plot = CustomPlot(Ev(df=df, x_name=category_cols),
                      ind_axx='catx',
                      ind_pltx='cat2',
                      ind_plty=None,
                      cat_column=['cat1'])

    show(plot._get_layout())




