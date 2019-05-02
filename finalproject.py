import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

def map_features(features,df,dict):
  for i in features:
    df = df.replace({i:dict})
  return df 

#convert move specialty
#1 stands for with specialty, 0 stands for without specialty
def convert_move_specialty(df):
  dict={}
  for i in df["move_specialty"]:
    if i=="U":
      dict.update({"U":0})
    else:
      dict.update({i:1})
  return dict

#replace missing values with an unique category
data=pd.read_csv("qudditch_training.csv")
columns_replace=["house","player_code","move_specialty"]
for column in columns_replace:
  data[column].replace("?","U",inplace=True)
data["gender"].replace("Unknown/Invalid","U",inplace=True)

df=pd.DataFrame(data=data)
#dict for mapping game move style
new_dict={'Steady':1,'No':0,'Up':0,'Down':0}
game_move_columns=["body_blow","checking","dopplebeater_defence","hawkshead_attacking_formation","no_hands_tackle","power_play","sloth_grip_roll","spiral_dive","starfish_and_stick","twirl","wronski_feint","zig-zag","bludger_backbeat","chelmondiston_charge","dionysus_dive","double_eight_loop","finbourgh_flick","reverse_pass","parkins_pincer","plumpton_pass","porskoff_ploy","transylvanian_tackle","woollongong_shimmy"]
df=map_features(game_move_columns,df,new_dict)
#convert target and  to 0 or 1
ordered_satisfaction = ["NO","YES"]
cat_dtype = pd.Categorical(ordered_satisfaction, ordered=True)
df["quidditch_league_player"]=df["quidditch_league_player"].astype(cat_dtype).cat.codes
#drop id_num,player_id,weight
df.drop(["id_num","player_id","weight"], axis=1,inplace=True)
ordered_satisfaction = ["No","Yes"]
cat_dtype = pd.Categorical(ordered_satisfaction, ordered=True)
df["snitch_caught"]=df["snitch_caught"].astype(cat_dtype).cat.codes
ordered_satisfaction = ["No","Ch"]
cat_dtype = pd.Categorical(ordered_satisfaction, ordered=True)
df["change"]=df["change"].astype(cat_dtype).cat.codes
#do one-hot endoing
df=pd.get_dummies(df, columns=["house","foul_type_id","game_move_id","penalty_id","player_code","player_type"])
#do label encoding
#version 1
'''
ordered_satisfaction = ["None", "Norm", ">7", ">8", ">200", ">300"]
cat_dtype = pd.api.types.CategoricalDtype(ordered_satisfaction, ordered=True)
df["snitchnip"]=df["snitchnip"].astype(cat_dtype).cat.codes
df["stooging"]=df["stooging"].astype(cat_dtype).cat.codes
'''
#version 2
ordered_satisfaction = ["None", "Norm",  ">200", ">300"]
cat_dtype = pd.Categorical(ordered_satisfaction, ordered=True)
df["snitchnip"]=df["snitchnip"].astype(cat_dtype).cat.codes
ordered_satisfaction = ["None", "Norm",  ">7", ">8"]
cat_dtype = pd.Categorical(ordered_satisfaction, ordered=True)
df["stooging"]=df["stooging"].astype(cat_dtype).cat.codes
df = df[df.gender != 'U']
ordered_satisfaction = ["Female","Male"]
cat_dtype = pd.Categorical(ordered_satisfaction, ordered=True)
df["gender"]=df["gender"].astype(cat_dtype).cat.codes
dict=convert_move_specialty(df)
df=map_features(["move_specialty"],df,dict)

norm_columns=["age","game_duration","num_game_moves","num_game_losses","num_practice_sessions","num_games_satout","num_games_injured","num_games_notpartof","num_games_won","snitchnip","stooging"]
scaler = preprocessing.MinMaxScaler()
for i in norm_columns:
	df[i] = scaler.fit_transform(df[i].values.reshape(-1,1))

df_target=pd.DataFrame(data=df["quidditch_league_player"])
df.drop(["quidditch_league_player"], axis=1,inplace=True)
'''
x=df.values
y=df_target.values
x=StandardScaler().fit_transform(x)
covar_matrix = PCA(n_components = len(df.columns))
covar_matrix.fit(x)
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)   
print (var)
'''
array=df.values
array = StandardScaler().fit_transform(array)
pca = PCA(n_components=len(df.columns)-9)
array_new = pca.fit_transform(array)
df_new=pd.DataFrame(array_new)
df_new.insert(len(df_new.columns),"quidditch_league_player", df_target)
df_new.to_csv("a.csv",index=False)
#df.insert(len(df.columns),"quidditch_league_player", df_target)
#df.to_csv("a.csv",index=False)
#df_target.to_csv("b.csv",index=False)

