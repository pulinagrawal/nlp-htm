from htm.bindings.sdr import SDR

''' A decorator function to convert a binary vector in a list to a SDR object '''
def to_sdr(n_columns, n_cells):
  def to_sdr(func):
    def wrapper(*args, **kwargs):
      result = func(*args, **kwargs)
      sdr = SDR(n_columns*n_cells)
      sdr.sparse = result
      return sdr
    return wrapper
  return to_sdr