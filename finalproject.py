import csv
import pandas as pd 
#replace missing values with an unique category
data=pd.read_csv("qudditch_training.csv")
columns_replace=["house","player_code","move_specialty"]
for column in columns_replace:
	data[column].replace("?","U",inplace=True)

data["gender"].replace("Unknown/Invalid","U",inplace=True)
#drop id_num,player_id,weight 
df=pd.DataFrame(data=data)
df.drop(["id_num","player_id","weight"], axis=1,inplace=True)
print(df)
df.to_csv("a.csv",index=False)

