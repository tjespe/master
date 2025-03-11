# %%
import pandas as pd
import numpy as np


def apply_fredmd_transform(series, transform_code):
    """
    Apply the recommended FRED-MD transform to a pandas Series.
    transform_code is an integer 1-7 as described in the FRED-MD documentation:
      1 -> Level (no transform)
      2 -> First difference
      3 -> Second difference
      4 -> Natural log
      5 -> Log first difference
      6 -> Log second difference
      7 -> Other special transforms (rarely used; sometimes log(1+x))
    """
    if transform_code == 1:
        # Level (no transform)
        return series
    elif transform_code == 2:
        # First difference
        return series.diff()
    elif transform_code == 3:
        # Second difference
        return series.diff().diff()
    elif transform_code == 4:
        # Log
        return np.log(series)
    elif transform_code == 5:
        # Log first difference
        return np.log(series).diff()
    elif transform_code == 6:
        # Log second difference
        return np.log(series).diff().diff()
    elif transform_code == 7:
        # Some special transformation, e.g. log(1 + x). Adjust as needed.
        return np.log1p(series).diff()  # as an example
    else:
        # If unknown code, just return as-is or raise an error
        return series


# %%
def get_fred_md():
    """
    Read the FRED-MD dataset and apply the recommended transformations.
    """
    # %%
    # ---------------------------------------------------------
    # STEP 1: READ HEADER + TRANSFORM CODES
    file_path = "data/fred_md.csv"  # or your filename
    with open(file_path, "r") as f:
        lines = f.read().splitlines()

    header_line = lines[0]
    transform_line = lines[1]

    # Parse the header into a list of column names
    column_names = header_line.split(",")

    # Parse the transform codes (dropping the first "Transform:" token)
    transform_items = transform_line.split(",")
    # The first item in transform_line is "Transform:", so skip it
    transform_codes = transform_items[
        1:
    ]  # this should align with the columns after "sasdate"

    # We expect len(transform_codes) == len(column_names) - 1
    # because sasdate usually won't have a transform code
    # but let's store them in a dict keyed by column name, skipping sasdate
    transform_map = {}
    for col_name, tcode in zip(column_names[1:], transform_codes):
        # Convert tcode to an integer, if possible
        # If it fails, you can handle or set to 1 by default
        try:
            transform_map[col_name] = int(tcode)
        except ValueError:
            transform_map[col_name] = 1  # or handle differently
    transform_map

    # %%
    # ---------------------------------------------------------
    # STEP 2: READ THE MAIN DATA (SKIPPING THE SECOND LINE)

    # We'll load the CSV again with pandas, telling it to skip row #1 (zero-based)
    # so that the second line (transform line) doesn't appear in the dataframe
    df = pd.read_csv(
        file_path,
        skiprows=[1],  # skip the transform line
        parse_dates=["sasdate"],  # parse sasdate as a datetime
        infer_datetime_format=True,
    )

    df.set_index("sasdate", inplace=True)  # set date as index if you like
    df.index.name = "Date"
    df

    # %%
    # ---------------------------------------------------------
    # STEP 3: APPLY TRANSFORMATIONS

    # Create a new dataframe to hold transformed data
    df_transformed = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col in transform_map:
            # Apply the recommended transform
            tcode = transform_map[col]
            df_transformed[col] = apply_fredmd_transform(df[col], tcode)
        else:
            # e.g. for sasdate, or any column not in transform_map
            df_transformed[col] = df[col]
    df_transformed

    # %%
    # ---------------------------------------------------------
    # STEP 4: DROP ROWS WITH NULL VALUES
    # Find first row where all values are non-null and drop any rows before it
    first_valid_row = df_transformed.notnull().all(axis=1).idxmax()
    df_transformed = df_transformed.loc[first_valid_row:]
    df_transformed

    # %%
    return df_transformed
