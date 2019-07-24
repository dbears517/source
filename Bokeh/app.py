# -*- coding: utf-8 -*-

import numpy as np
from bokeh.plotting import figure
from bokeh.io import output_file, show

fg = figure(x_axis_label="x_axis", y_axis_label="y_axis")

x=[1,2,3,4]
y=[1,2,3,4]

fg.circle(x,y)

output_file('sample_plot.html')

show(fg)

