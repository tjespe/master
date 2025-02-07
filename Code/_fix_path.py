import os

# Code for adding Git LFS to the PATH so that git push etc. works when used in a notebook
os.environ["PATH"] = (
    os.environ["PATH"]
    + ";"
    + "C:\\Users\\"
    + os.environ["USERNAME"]
    + "\\AppData\\Local\\Programs\\Git LFS"
)
