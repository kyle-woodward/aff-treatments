"""
Script for defining functions for parsing csv data in earth engine
"""

import ee
ee.Initialize()

def parse_txt(blob: ee.Blob, delim: str = ",", qualifier: str = '"') -> ee.Dictionary:
    """Function to parse csv objects from Cloud storage on EE
    Expects data be formatted with no missing values and the first row be the column headers
    args:
        blob (ee.Blob): Blob object from cloud storage to parse
        delim (str): string used to differentiate colummns. default = ,
        qualifier (str): string used to wrap values in column. default - "
    returns:
        ee.Dictionary: dictionary representation of csv where keys are column names
            and values are list of values
    """

    def clean_string(x: ee.ComputedObject) -> ee.String:
        """Closure function used to clean up the"""
        # force the value to be string
        x = ee.String(x)
        # strip off the qualifier value and trim any white space
        return x.replace(f"^({qualifier})", "").replace(f"({qualifier})$", "").trim()

    # get ee.String from blob
    txt_str = blob.string()

    # split the lines
    # do we need to add in option for specifying line terminus?
    lines = txt_str.split("\n")
    # get the first row and set aside as header values
    header = ee.String(lines.get(0)).split(delim).map(clean_string)

    # loop over the rows and split into columns
    data = lines.slice(1).map(lambda x: ee.String(x).split(delim).map(clean_string))

    # create a dictionary from key/value (i.e. header/col) pairs
    # columns come out formatted as [[row],[row],..] so use zip to convert to [[col],[col],..]
    txt_dict = ee.Dictionary.fromLists(header, data.unzip())

    return txt_dict


def to_numeric(list: ee.List) -> ee.List:
    """Helper function to convert parsed col list from string to numeric values
    args:
        list (ee.List): list to convert to numeric values
    returns
        ee.List: list containing numeric values
    """
    return list.map(lambda x: ee.Number.parse(ee.String(x)))