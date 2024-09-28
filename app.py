import pandas as pd


# function to read the dat
def load_data():
    file_name = "./data/zomato_df_final_data.csv"
    df = pd.read_csv(file_name)
    print(f"successfully loaded the data. The shape of the data is {df.shape}")
    

def main():
    
    # load data
    load_data()
    
    
main()