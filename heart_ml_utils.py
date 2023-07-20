# Custom functions for heart disease detection project


def drop_data(df, col, value):
    """
    Description: deletes observations with a given value in a given column of a given dataframe
    Inputs: df = data frame from which to remove observations
            col = the column to be searched for the value
            value =  the value that must be present in order for the column to be deleted
    Outputs: new_df = data frame with problem observations removed
    """
    new_df = df.loc[df[col] != value]
    return new_df


def string_to_Na(df, string):
    """
    Description: converts cells with a given string value into None in a given data frame
    Inputs: df =  data frame to change
            string = string a cell must have to be converted into None
    Outputs: df = updated data frame
    """
    for col in df.select_dtypes(exclude='number').columns:
        df.loc[df[col]==string, col]=None
    return df
