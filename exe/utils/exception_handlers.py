import os
import sys
import traceback

from utils.plotting_functions import plot


def save_snapshot(vm, fig, ax, _frames_dir, index, cbar_zero='hexagon'):
    """ Saves simulation snapshot while taking care of syntax errors """
    
    # update plot
    plot(vm, fig=fig, ax=ax, update=True, cbar_zero=cbar_zero)

    # save frame
    while True:
        try:
            fig.savefig(os.path.join(_frames_dir, "%05d.png" % index))
            break
        
        except SyntaxError:
            # dirty fix to "SyntaxError: not a PNG file" with multiple matplotlib instances
            print(traceback.format_exc(), file=sys.stderr)
            pass
