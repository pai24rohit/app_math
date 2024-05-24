
def derive(f, x, h=0.0001):
    fx_plus_h = f(x + h)
    fx_minus_h = f(x - h)
    return (fx_plus_h - fx_minus_h) / (2 * h)
 # TODO: implement this function
