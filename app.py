import pandas as pd
import time
import matplotlib.pyplot as plt
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

porterStemmer = PorterStemmer()

# Removing the stop words
# from gensim.parsing.preprocessing import remove_stopwords
# print(remove_stopwords("Restaurant had a really good service!!"))



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
    '''use this function to avoid imbalance in data'''
    df_p = df[ df['target'] == 1 ].head(n)
    df_ng = df[ df['target'] == -1 ].head(n)
    df_n = df[ df['target'] == 0 ].head(n)
    res = pd.concat([df_ng, df_n, df_p])
    # print( res.value_counts() )
    return res

def stem( df ):
    '''stemming the words'''
    def s_( x ):
        return ' '.join( [ porterStemmer.stem( word ) for word in x.split(' ') ] )
    df['stem'] = df['sentence'].apply( s_ )
    return df

def split_data(df, ):
    '''split the data into train, test and randomise the data.'''
    x_train, x_test, y_train, y_test = train_test_split(df,
                                                        df['target'],
                                                        shuffle=True,
                                                        test_size=.3,
                                                        random_state=42
                                                        )
    
    # reset index for data
    x_train = x_train.reset_index()
    x_test = x_test.reset_index()
    # 
    y_train = y_train.to_frame()
    y_train = y_train.reset_index()
    # 
    y_test = y_test.to_frame()
    y_test = y_test.reset_index()
    return x_train, x_test, y_train, y_test
    
    

df = get_data()
df = get_top(df, 10000 )
df = stem( df )
x_train, x_test, y_train, y_test = split_data(df)

start_time = time.time()



# 
# def f():
#     input_ = input("Enter value: ")
#     if len( input_ ) != 4:
#         return f()
#     else:
#         return input_
    
# def f2( x ):
#     print("this is function 2", x)
        
# d = f()
# print( "out put return from f function here....", d )
# f2(d)