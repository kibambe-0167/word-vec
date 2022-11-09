import pandas as pd
import matplotlib.pyplot as plt



def label_( x ):
    x =  ' '.join( x.split() )
    x = x.lower()
    if x == 'positive': return 1
    elif x == 'negative': return -1
    else: return 0

def get_data():
    data = pd.read_csv("./allTweetsClean.csv", header=0)
    data.rename({" sentiment": "sentiment"}, axis=1, inplace=True )
    data[ data['sentiment'].str.len() == 1 ] = ' neutral'
    data['target'] = data['sentiment'].apply( label_ )
    # print( data )
    
    plt.figure()
    pd.value_counts( data.target ).plot.bar(title="Sentiment distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of rows")
    # plt.show()
    return data

def get_top( df, n = 5000 ):
    '''use this function to avoid inbalance in data'''
    df_p = df[ df['target'] == 1 ].head(n)
    df_ng = df[ df['target'] == -1 ].head(n)
    df_n = df[ df['target'] == 0 ].head(n)
    res = pd.concat([df_ng, df_n, df_p])
    # print( res.value_counts() )
    return res
    

df = get_data()
df = get_top(df, 10000 )