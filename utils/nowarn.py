import warnings
def no_warning(*args, **kwargs):
    pass

warnings.showwarning = no_warning