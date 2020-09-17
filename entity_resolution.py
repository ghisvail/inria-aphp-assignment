import pandas as pd
from functools import lru_cache


STATES = {'act', 'nsw', 'nt', 'qld', 'sa', 'tas', 'vic', 'wa'}


@lru_cache(maxsize=1)
def read_state_postcode():
    return pd.read_csv("state_postcode.csv").convert_dtypes().astype({
        "postcode_min_range": int, "postcode_max_range": int})


def get_postcode_validator():
    state_postcode = read_state_postcode()

    postcode_ranges = pd.arrays.IntervalArray.from_arrays(
        left=state_postcode.postcode_min_range,
        right=state_postcode.postcode_max_range,
        closed="both",
    )

    return lambda p: postcode_ranges.contains(int(p)).any()


def get_state_postcode_validator():
    state_postcode = read_state_postcode()

    postcode_ranges_per_state = {
        state: pd.arrays.IntervalArray.from_arrays(
            left=postcode.postcode_min_range,
            right=postcode.postcode_max_range,
            closed="both",
        )
        for state, postcode
        in state_postcode.groupby(by="state").agg(tuple).iterrows()
    }

    return lambda s, p: postcode_ranges_per_state[s].contains(int(p)).any()


def drop_duplicated_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates(subset={"patient_id"}, keep=False)
        .set_index(keys="patient_id")
    )


def sanitize_street_number(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({"street_number": {0: pd.NA}})


def sanitize_date_of_birth(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        date_of_birth=pd.to_datetime(
            df.date_of_birth,
            format="%Y%m%d",
            errors="coerce",
        )
    )


def sanitize_suburb(df: pd.DataFrame) -> pd.DataFrame:
    # Detect where swapping of suburb with postocode occurred
    df_swapped = df.loc[df.suburb.str.contains(r"\d"), ["suburb", "postcode"]]

    # Perform swapping of suburb with postcode
    df_swapped["suburb"], df_swapped["postcode"] = df_swapped["postcode"], df_swapped["suburb"]
    
    # Correct for possible typos in postcode
    df_swapped.postcode = df_swapped.postcode.str.replace(r"[a-z]", "")
    
    df.update(df_swapped)

    return df


def sanitize_postcode(df: pd.DataFrame) -> pd.DataFrame:
    validate_postcode = get_postcode_validator()

    postcodes = df.postcode.dropna().unique()

    to_replace = {p: pd.NA for p in postcodes if not validate_postcode(p)}

    return df.replace({"postcode": to_replace})


def sanitize_state(df: pd.DataFrame) -> pd.DataFrame:
    from itertools import product
    from textdistance import damerau_levenshtein

    # List of valid codes for state
    states = STATES

    # Compute list of codes to match
    codes = set(df.state.dropna()) - states

    # Compute string similarity between all combinations of code and state
    df_distance = pd.Series(
        data=[damerau_levenshtein(c, s) for c, s in product(codes, states)],
        index=pd.MultiIndex.from_product([codes, states], names=["code", "state"]),
        name="distance",
    )

    # Keep unambiguous code to state matches
    code_to_state = dict(
        df_distance[df_distance == 1].index.to_frame()
        .drop_duplicates(subset="code", keep=False).index
    )
    
    # Fill ambiguous and unmatched codes to N/A
    codes_to_na = codes - set(code_to_state.keys())
    code_to_state.update({c: pd.NA for c in codes_to_na})

    return df.replace({"state": code_to_state})


def clean_state_with_postcode(df: pd.DataFrame) -> pd.DataFrame:
    validate = get_state_postcode_validator()

    df_postcode_state = df[["postcode", "state"]].dropna()

    where_incoherent = df_postcode_state.apply(
        lambda x: not validate(x.state, x.postcode),
        axis="columns",
    )

    df.loc[where_incoherent.index, "state"] = pd.NA

    return df


def infer_state_from_postcode(df: pd.DataFrame) -> pd.DataFrame:
    from itertools import product

    validate_state_postcode = get_state_postcode_validator()

    df_postcode = df[df.state.isna()].postcode.dropna()

    postcodes, states = df_postcode.unique(), STATES

    to_replace = pd.Series(
        data={
            postcode: state
            for postcode, state in product(postcodes, states)
            if validate_state_postcode(state, postcode)
        },
        name="state"
    )

    df_state = df_postcode.replace(to_replace).rename("state")

    df.update(df_state)

    return df


def dedup_patient(df: pd.DataFrame) -> pd.DataFrame:
    df["dedup_id"] = pd.Series([], dtype="Int64")
    return df


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(drop_duplicated_patient_id)
          .pipe(sanitize_date_of_birth)
          .pipe(sanitize_street_number)
          .pipe(sanitize_suburb)
          .pipe(sanitize_postcode)
          .pipe(sanitize_state)
          .pipe(clean_state_with_postcode)
          .pipe(infer_state_from_postcode)
          #.pipe(dedup_patient)
    )