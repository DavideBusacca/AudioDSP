#
# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.

# Adapted by Davide Busacca for PV

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import patches


def plot_zplane(z, p, k, subplot=None):
    """Plot the complex z-plane.
    """
    if subplot is None:
        subplot = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False, color='black', ls='dashed')
    subplot.add_patch(uc)

    # Plot the zeros and set marker properties
    t1 = subplot.plot(z.real, z.imag, 'go', ms=10, label='Zeros')
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0, markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = subplot.plot(p.real, p.imag, 'rx', ms=10, label='Poles')
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0, markeredgecolor='r', markerfacecolor='r')

    subplot.spines['left'].set_position('center')
    subplot.spines['bottom'].set_position('center')
    subplot.spines['right'].set_visible(False)
    subplot.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5
    subplot.axis('scaled')
    subplot.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]
    subplot.set_xticks(ticks)
    subplot.set_yticks(ticks)

    subplot.legend(loc='upper right')