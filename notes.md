I have removed 16 rows from breast-cancer-wisconsin.data due to missing data (some ?-n)

now there are 685 rows left

In wines dataset I replaced white with 1 and red with 0

Made all the label columns in data_binary be "labels"

df_  = pd.read_csv(r"datasets\breast-cancer-wisconsin.csv")
df_["labels"] = df_["labels"].replace({2: 0, 4: 1})