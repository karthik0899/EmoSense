def INFO(df):
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    info = []

    for col in df.columns:
        count_rows = len(df[col])
        NAN_values = df[col].isna().sum()
        percent = (NAN_values / count_rows) * 100
        data_type = type(df[col][0])
        col_type = df[col].dtype
        if col_type not in [int,float]:
            column_type = "Categorical"
            Max = "Not Applicable"
            Min = "Not Applicable"
        else:
            column_type = "Numirical"
            Max = max(df[col])
            Min = min(df[col])
        try:
            n_uniques =df[col].nunique()
            ratio = count_rows/n_uniques
        except:
            n_uniques = "Not Applicable"
            ratio = "Not Applicable"
        info.append([col,data_type,column_type, count_rows, NAN_values, percent,n_uniques,ratio,Max, Min])

    col_info_df = pd.DataFrame(info, columns=['Column','Data Type','Column Type', 'count_rows', 'Missing', 'Percent Missing','Number of Uniques','Ratio of uniqus','Max','Min'])

    return col_info_df
