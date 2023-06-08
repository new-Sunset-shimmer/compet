import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("/home/yangcw/cw/bookdatas/test_data.csv")
df1 = pd.read_csv("clear_test.csv")
# df1 = pd.read_csv("caption_data_train1.csv")
x1,x2 = train_test_split(y, shuffle=False)
def delete_dup(x):
    if x in ['book','cover','the','a','close','up','person','of','the','with']:
        return ''
    return x
    

# df['caption']=df['caption'].apply(lambda x: " ".join(set([delete_dup(y.strip()) for y in str(x).split(' ') ])))
# df1['caption']=df1['caption'].apply(lambda x: " ".join(set([delete_dup(y.strip()) for y in x.split(' ') ])))

df3 = pd.concat([df,df1],keys=['id','caption'])
df.to_csv("./clear_test.csv",index=False)
# df.to_csv("./clear_train.csv",index=False)