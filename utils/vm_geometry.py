

def hexagon_volume(rgrid):
    """ Computes volume of regular hexagon with side length rgrid  """

    return (3**2 / 2) * rgrid**3


def hexagon_area(rgrid):
    """ Computes area of regular hexagon with side length rgrid """

    return (3**(3/2) / 2) * (rgrid)**2

def hexagon_side(V0):
    """ Computes side length of regular hexagon fm volume """

    return ((2 / 3**2) * V0) ** (1/3)
