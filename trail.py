import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

df_drop = pd.read_csv("rulesDataset.csv")

# df_drop = df_drop[['Residence_city','Civil_status','STEM_subjects','H_subjects','Dropout']].copy()
print(df_drop.columns)

df = df_drop.copy()
df.fillna(0, inplace=True)

encode = ['Residence_city','Civil_status','State','Province','Desired_program','Father_level','Mother_level']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df['Dropout'] = np.where(df['Dropout']=='Yes', 1, 0)
X = df.drop('Dropout', axis=1)
Y = df['Dropout']

clf = RandomForestClassifier()
clf.fit(X, Y)
print(X.columns)

# new_input = [[0,0,1,0,0,1,0,49.0800,45.2000]]
# new_output = clf.predict(new_input)
# print(new_input, new_output)

pickle.dump(clf, open('dropout_used_all.pkl', 'wb'))


# # var_1 to be input given by user.
# var_1 = df.iloc[:1]
# var_1 = pd.DataFrame(var_1, columns=df.columns)
# # print(var_1)

# a = pd.concat([df,var_1], axis=0)
# print(df.shape, var_1.shape)
# # print(a)

# df_num_fet = a.select_dtypes(object)
# # print(df_num_fet)
# b = pd.get_dummies(df_num_fet, drop_first=True)
# b.reset_index(inplace = True, drop=True)
# print(b)

# df_num_num = a.select_dtypes(np.number)
# df_num_num.reset_index(inplace=True, drop=True)
# final_data = pd.concat([df_num_num,b], axis=1)
# print(final_data)

# some = final_data.iloc[-1]
# print(some.values)