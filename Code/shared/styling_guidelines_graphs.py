from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

# === Seaborn base theme (for litt ekstra polish) ===
sns.set_theme(context="notebook", style="whitegrid")

# === LaTeX Fonts === # TODO uncomment to get LaTeX font
# mpl.rcParams.update({
#    'text.usetex': True,
#    'font.family': 'serif',
#    'font.serif': ['Computer Modern Roman'],
#    'axes.unicode_minus': False  # Sørger for at minus-tegn vises korrekt
# })

# === Plot Størrelser og DPI ===
mpl.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# === Tekst og Fontstørrelser ===
mpl.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.size": 12,
    }
)

# === Akser og Rutenett ===
mpl.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.loc": "best",
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)

# === Gråtoneskala-palett ===
colors = {
    "primary": "#2166ac",
    "secondary": "#d6604d",
    "accent": "#f4a582",
    "muted": "#fddbc7",
    "dark": "#d1e5f0",
    "light": "#92c5de",
    "background": "#4393c3",
    "highlight": "#b2182b",
}
# Sett farge-syklusen i matplotlib
color_cycle = list(colors.values())
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_cycle)
