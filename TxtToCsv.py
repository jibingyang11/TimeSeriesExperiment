import pandas as pd
input_file='traffic.txt'
output_file='traffic.csv'
df=pd.read_csv(input_file,sep=',')
df.to_csv(output_file,index=False)
