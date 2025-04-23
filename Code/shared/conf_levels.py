# %%
def format_cl(cl):
    return f"{100*cl:1f}".rstrip("0").rstrip(".")


def get_VaR_level(cl):
    return 1 - (1 - cl) / 2
