import pandas as pd
from tqdm import tqdm, trange,tnrange,tqdm_notebook
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home2/yangcw/clear_train_v3.csv")
# df1 = pd.read_csv("/home2/yangcw/clear_val_v4.csv")
# df3 = pd.concat([df,df1],keys=['id','caption','Title','label'])
# print(len(df3))
print(len(df))
def reverser(x):
    if x==True:return False 
    return True
df3 = df['Title'].duplicated()
df3 = [reverser(x) for x in df3 ]
# # # df.to_csv("./clear_train.csv",index=False)

# # # df2 = pd.read_csv("./clear_train.csv")
# # # df3 = pd.read_csv("/home2/yangcw/bookdatas/test_data.csv")
# # # df4 = pd.read_csv("/home2/yangcw/bookdatas/train_data.csv")
df3 = df[df3]
print(len(df3))
x1,x2 = train_test_split(df3, shuffle=True,random_state=19,test_size=0.2)
x1.to_csv('./clear_train_v5.csv',index=False)   
x2.to_csv('./clear_val_v5.csv',index=False)  
# df_test = pd.DataFrame({"id" : df4['id'],"caption" : df2['caption'],"Title" : df4['Title'],"label" : df4['label']})
# df_test.to_csv('./clear_train_v2.csv',index=False)   

# df = pd.read_csv("/home/yangcw/cw/bookdatas/caption_data.csv")
# df = pd.read_csv("clear_test_v2.csv")
# df1 = pd.read_csv("clear_train_v2.csv")
# def delete_dup(x):
#     if x in ['book','cover','the','a','close','up','person','of','the','with']:
#         return ''
#     return x
    

# df['caption']=df['caption'].apply(lambda x: " ".join(str(y.strip()) for y in str(x).split(' ') ))
# df1['caption']=df1['caption'].apply(lambda x: " ".join(str(y.strip()) for y in str(x).split(' ')))

# # df = pd.concat([df,df1],keys=['id','caption'])
# df.to_csv("./clear_test_v3.csv",index=False)
# df1.to_csv("./clear_train_v3.csv",index=False)