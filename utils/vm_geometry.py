

def hexagon_volume(rhex):
    """ Computes volume of regular hexagon with side length rhex  """

    return (3**2 / 2) * rhex**3


def hexagon_area(rhex):
    """ Computes area of regular hexagon with side length rhex """

    return (3**(3/2) / 2) * (rhex)**2

def hexagon_side(V0):
    """ Computes side length of regular hexagon from volume """

    return ((2 / 3**2) * V0) ** (1/3)
