"""Processes the phenotypic data and makes mappings from patient ID to pertinent phenotypic features"""

from sys import argv
from pandas import concat, DataFrame, get_dummies, factorize, read_csv, Series
from numpy import concatenate, ndarray, nanmin, nanmax, isnan
from pickle import dump
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer
# noinspection PyUnresolvedReferences,PyProtectedMember
from sklearn.impute import IterativeImputer, SimpleImputer

from handler.utils import (
    PATIENT_ID_COL_NAME, NUMERIC_COL_TYPE, normalize, DATASET_PATH, COL_TYPES_PATH, get_del_col,
    RAW_PHENOTYPES_DATA_PATH, PTIDS_PATH
)

PHENOTYPES_KEY: str = 'phenotypes'
KNOWN_VAL_THRESHOLD: float = 0.8
NOMINAL_COL_TYPE: str = 'nominal'
MIN_CATEGORY_SIZE: int = 10


def handle():
    """Main method of this module"""

    # Load in the raw data set and the table that indicates the data type of each column
    cohort: str = argv[2]
    data_path: str = RAW_PHENOTYPES_DATA_PATH
    data_set: DataFrame = read_csv(data_path, low_memory=False)
    col_types_path: str = 'MergeTables/ToCSV/col-types.csv'
    col_types: DataFrame = read_csv(col_types_path, low_memory=False)
    ptids_path: str = PTIDS_PATH.format(cohort)
    ptids: DataFrame = read_csv(ptids_path)

    data_set, col_types = get_mappings(data_set=data_set, col_types=col_types, cohort=cohort, ptids=ptids)

    # Process the nominal columns
    nominal_data, nominal_cols = clean_nominal_data(data_set=data_set, data_types=col_types)

    # Process the numeric columns
    numeric_data: DataFrame = clean_numeric_data(
        data_set=data_set, data_types=col_types, nominal_data=nominal_data, nominal_cols=nominal_cols
    )

    # Combine the processed nominal data with the processed numeric data
    data_set: DataFrame = concat([ptids, numeric_data, nominal_data], axis=1)

    phenotypes_path: str = DATASET_PATH.format(cohort, PHENOTYPES_KEY)
    data_set.to_csv(phenotypes_path, index=False)
    col_types_path: str = COL_TYPES_PATH.format(cohort, PHENOTYPES_KEY)
    col_types.to_csv(col_types_path, index=False)


def get_mappings(data_set: DataFrame, col_types: DataFrame, cohort: str, ptids: DataFrame) -> tuple:
    """Creates mappings from patient ID to other pertinent features, removing rows that don't have them from the data"""

    # TODO: make this work for ANM too

    cdr_feat: str = 'CDGLOBAL'
    feats_to_map: list = [cdr_feat, 'PTGENDER']
    ptids: Series = ptids[PATIENT_ID_COL_NAME]
    data_set: DataFrame = data_set.set_index(PATIENT_ID_COL_NAME)
    data_set: DataFrame = data_set.loc[ptids]

    for feat in feats_to_map:
        # Remove rows in which the current feature is unknown
        data_set: DataFrame = data_set[data_set[feat].notna()]

    combine_highest_cdr_cat_with_second_highest(data_set=data_set, cdr_feat=cdr_feat)
    remove_unacceptable_cols(data=data_set, col_types=col_types)
    remaining_feats: list = list(data_set)
    col_types: DataFrame = col_types[remaining_feats].copy()

    # Create a mapping of patient IDs to the current feature
    ptid_col: Series = Series(data_set.index)
    ptid_to_feat: dict = {}

    for feat in feats_to_map:
        feat_col: Series = data_set[feat].copy()

        for ptid, val in zip(ptid_col, feat_col):
            ptid_to_feat[ptid] = val

        ptid_to_feat_path: str = 'processed-data/feat-maps/{}/{}.p'.format(cohort, feat.lower())

        with open(ptid_to_feat_path, 'wb') as f:
            dump(ptid_to_feat, f)

    return data_set, col_types


def combine_highest_cdr_cat_with_second_highest(data_set: DataFrame, cdr_feat: str):
    """Combines the highest CDR category with the second highest category"""

    cdr_col: Series = data_set[cdr_feat].copy()
    max_cdr: float = cdr_col.max()
    cdr_col[cdr_col == max_cdr] = max_cdr - 1
    data_set[cdr_feat] = cdr_col


def remove_unacceptable_cols(data: DataFrame, col_types: DataFrame):
    """Removes columns from the current data set that only have one unique value or too few known values"""

    for col_name in list(data):
        col: Series = data[col_name]
        known_ratio: float = col.notna().sum() / len(col)

        if len(col.unique()) == 1:
            del data[col_name]
            continue

        if known_ratio < KNOWN_VAL_THRESHOLD:
            del data[col_name]
            continue

        if col_types[col_name][0] == NOMINAL_COL_TYPE:
            categories: list = col.unique()

            for category in categories:
                category: Series = col[col == category]

                if len(category) < MIN_CATEGORY_SIZE:
                    del data[col_name]
                    break


def get_cols_by_type(data_set: DataFrame, data_types: DataFrame, col_type: str) -> tuple:
    """Gets the columns and column names of a given type"""

    col_bools: Series = data_types.loc[0] == col_type
    cols: Series = data_types[col_bools.index[col_bools]]
    cols: list = list(cols)
    data: DataFrame = data_set[cols]
    return data, cols


def clean_nominal_data(data_set: DataFrame, data_types: DataFrame):
    """Processes the nominal data"""

    nominal_data, nominal_cols = get_cols_by_type(data_set=data_set, data_types=data_types, col_type=NOMINAL_COL_TYPE)

    # Impute unknown nominal values
    imputer: SimpleImputer = SimpleImputer(strategy='most_frequent', verbose=2)
    # noinspection PyUnresolvedReferences
    nominal_data: ndarray = nominal_data.to_numpy()
    nominal_data: ndarray = imputer.fit_transform(nominal_data)
    nominal_data: DataFrame = DataFrame(nominal_data, columns=nominal_cols)

    # Replace commas in nominal values with periods to avoid parsing issues later
    for col_name in list(nominal_data):
        col: Series = nominal_data[col_name]

        for idx, val in zip(nominal_data.index, col):
            assert nominal_data[col_name][idx] == val

            if ',' in str(val):
                val: str = val.replace(',', '.')
                nominal_data[col_name][idx] = val

    for col in list(nominal_data):
        col: Series = nominal_data[col]

        for val in col:
            assert ',' not in str(val)

    return nominal_data, nominal_cols


def clean_numeric_data(
        data_set: DataFrame, data_types: DataFrame, nominal_data: DataFrame, nominal_cols: list, impute_seed=0,
        max_iter=50, n_nearest_features=150
) -> DataFrame:
    """Processes the numeric data"""

    # One hot encode the nominal values for the purpose of imputing unknown real values with a more sophisticated method
    one_hot_nominal_data: DataFrame = get_dummies(nominal_data, columns=nominal_cols, dummy_na=False)

    # Get the numeric columns and column names
    numeric_data, numeric_cols = get_cols_by_type(data_set=data_set, data_types=data_types, col_type=NUMERIC_COL_TYPE)

    # Normalize the numeric columns
    numeric_data: DataFrame = normalize(df=numeric_data)

    n_numeric_cols: int = numeric_data.shape[1]

    # Combine the nominal columns with the numeric so the nominal columns can be used in the imputation
    data_to_impute: ndarray = concatenate([numeric_data.to_numpy(), one_hot_nominal_data.to_numpy()], axis=1)

    # Impute missing numeric values
    imputer: IterativeImputer = IterativeImputer(
        verbose=2, random_state=impute_seed, max_iter=max_iter, max_value=nanmax(data_to_impute),
        min_value=nanmin(data_to_impute), n_nearest_features=n_nearest_features
    )

    imputed_data: ndarray = imputer.fit_transform(data_to_impute)

    assert not isnan(imputed_data.mean())

    # Separate the imputed numeric columns from the nominal columns that helped impute
    numeric_data: ndarray = imputed_data[:, :n_numeric_cols]
    numeric_data: DataFrame = DataFrame(data=numeric_data, columns=numeric_cols)
    return numeric_data
