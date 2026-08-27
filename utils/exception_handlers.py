import os
import sys
import traceback

from utils.vm_plotting import plot_frame


def save_frame(vm, fig, ax, _frames_dir, index, cbar_zero='hexagon', hmax=14):
    """ Saves simulation snapshot while taking care of syntax errors """
    
    # update plot
    plot_frame(vm, fig=fig, ax=ax, update=True, cbar_zero=cbar_zero, hmax=hmax)

    # save frame
    while True:
        try:
            fig.savefig(os.path.join(_frames_dir, "%05d.png" % index))
            break
        
        except SyntaxError:
            # dirty fix to "SyntaxError: not a PNG file" with multiple matplotlib instances
            print(traceback.format_exc(), file=sys.stderr)
            pass
