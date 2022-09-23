import matplotlib.pyplot as plt
import warnings
import numpy as np
import copy


def figsizepx(width: int, height: int, dpi: int = 100) -> tuple:
    """Calculate figsize from pixels of width and height.

    Parameters
    ----------
    width: int
        width of plot axis.

    height: int
        height of plot axis.

    dpi: int, optional
        Specify the figure's dpi. Default to 100.

    Returns
    -------
    figsize: tuple
        figsize.
    """
    ax_w_inch = width / dpi
    ax_h_inch = height / dpi
    ax_margin_inch = (0.5, 0.5, 0.5, 0.5)  # Left,Top,Right,Bottom [inch]
    fig_w_inch = ax_w_inch + ax_margin_inch[0] + ax_margin_inch[2]
    fig_h_inch = ax_h_inch + ax_margin_inch[1] + ax_margin_inch[3]
    return tuple(fig_w_inch, fig_h_inch)


def set_rc(**kwargs):
    if kwargs.get('fontsize') is not None and isinstance(kwargs.get('fontsize'), int):
        plt.rcParams['font.size'] = kwargs.get('fontsize')
    else:
        plt.rcParams['font.size'] = 20

    if kwargs.get('fontname') is not None and isinstance(kwargs.get('fontname'), str):
        plt.rcParams['font.family'] = kwargs.get('fontname')
    else:
        plt.rcParams['font.family'] = "Times New Roman"

    if kwargs.get('xtick_direction') is not None and isinstance(kwargs.get('xtick_direction'), str):
        plt.rcParams['xtick.direction'] = str(kwargs.get('xtick_direction'))
    else:
        plt.rcParams['xtick.direction'] = "in"
    if kwargs.get('ytick_direction') is not None and isinstance(kwargs.get('ytick_direction'), str):
        plt.rcParams['ytick.direction'] = str(kwargs.get('ytick_direction'))
    else:
        plt.rcParams['ytick.direction'] = "in"

    if kwargs.get('no_warnings') is not None and type(kwargs.get('no_warnings')) == bool:
        if kwargs.get('no_warnings'):
            warnings.simplefilter('ignore')
    else:
        warnings.simplefilter('ignore')


def __general_kwargs(**kwargs):
    if kwargs.get('color') is None:
        kwargs['color'] = 'k'
    if kwargs.get('linewidth') is None:
        kwargs['linewidth'] = 1.15
    if kwargs.get('xlabel') is None:
        kwargs['xlabel'] = 'Time [s]'
    if kwargs.get('ylabel') is None:
        kwargs['ylabel'] = 'Amplitude [mV]'
    if kwargs.get('cmap') is None:
        kwargs['cmap'] = 'viridis'
    if kwargs.get('shading') is None:
        kwargs['shading'] = 'auto'
    if kwargs.get('alpha') is None:
        kwargs['alpha'] = 0.45
    if kwargs.get('grid') is None:
        kwargs['grid'] = False
    return kwargs


def close_all():
    plt.close()


def plot1d_journal(x: any, fs: int = 128, **kwargs):
    """Plot x in line with format of journal.

    Parameters
    ----------
    x: any
        1D-data points.

    fs: int, optional
        Sampling frequency. Defaults to 128.

    kwargs: properties, optional

    """
    set_rc(**kwargs)
    args = __general_kwargs(**kwargs)
    if args.get('label1') is None:
        args['label1'] = ''

    plt.figure(figsize=(10, 3))
    ax = plt.axes([.092, .22, .887, .75])
    if args.get('axis') is None:
        args['axis'] = np.linspace(0, len(x)/fs, len(x))
    ax.plot(args['axis'], x, args['color'], linewidth=args['linewidth'],
            label=args['label1'])

    if args.get('xlim') is None:
        args['xlim'] = (args['axis'][0], args['axis'][-1])
    ax.set_xlim(args['xlim'])

    ax.set_xlabel(args['xlabel'])
    ax.set_ylabel(args['ylabel'])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.minorticks_on()
    if kwargs.get('grid'):
        plt.grid()

    if args.get('x2') is not None:
        if args.get('color2') is None:
            args['color2'] = 'r'
        if args.get('linewidth2') is None:
            args['linewidth2'] = copy.copy(args['linewidth'])
        if args['label2'] is None:
            args['label2'] = ''
        if args.get('legend') is None:
            args['legend'] = True

        ax.plot(np.linspace(0, len(x)/fs, len(x)), args['x2'], args['color2'], linewidth=args['linewidth2'],
                label=args['label2'])
        if args['legend']:
            plt.legend(loc='lower right',
                       fancybox=False, edgecolor="black",
                       borderpad=0.35, fontsize=17)


def plot2d_journal(img: any, x: any = None, y: any = None, **kwargs):
    """Create a pseudo-color plot in line with format of journal.

    Parameters
    ----------
    img: any
        A scalar 2-D array. The values will be color-mapped.

    x: any
        1-D arrays or column vectors as coordinate x.

    y: any
        1-D arrays or column vectors as coordinate y.

    kwargs: properties, optional
    """

    if x is None:
        x = np.linspace(0, img.shape[1], img.shape[1])
    if y is None:
        y = np.linspace(0, img.shape[0], img.shape[0])

    assert len(img.shape) == 2, "shape of argument img must be 2 dim."
    assert len(x) == img.shape[1] and len(y) == img.shape[0], "x or y dimensions did not match img."

    set_rc(**kwargs)
    args = __general_kwargs(**kwargs)

    if args.get('clabel') is None:
        args['clabel'] = ''

    plt.figure(figsize=(10, 3.5))
    ax = plt.axes([.092, .184, .887, .57])
    cax = plt.gcf().add_axes([.092, .77, .887, .0185])
    mappable = ax.pcolor(x, y, img,
                         shading='auto',
                         cmap=args['cmap']
                         )
    ax.set_xlabel(args['xlabel'])
    ax.set_ylabel(args['ylabel'])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.f'))
    cax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.minorticks_on()

    plt.colorbar(mappable=mappable, cax=cax, orientation='horizontal')
    cax.tick_params(bottom=False, top=False, direction='out')
    cax.xaxis.set_ticks_position('top')
    plt.title(args['clabel'], fontsize=18)
