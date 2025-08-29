import os
from cells.plot import plot



def save_snapshot(vm, fig, ax, _frames_dir, index):
    # update plot
    plot(vm, fig=fig, ax=ax, update=True)

    # save frame
    while True:
        try:
            fig.savefig(os.path.join(_frames_dir, "%05d.png" % index))
            break
        
        except SyntaxError:
            # dirty fix to "SyntaxError: not a PNG file" with multiple matplotlib instances
            print(traceback.format_exc(), file=sys.stderr)
            pass
