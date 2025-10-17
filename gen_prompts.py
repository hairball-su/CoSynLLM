import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pandas as pd 
import pickle

# # # 读取 Excel 文件中的数据
excel_data = pd.read_excel('ALMANAC_drug_smiles.xlsx')

# #deepseek
selected_columns = excel_data[['pubchemid', 'deepseekv3']]
data_dict = selected_columns.set_index('pubchemid')['deepseekv3'].to_dict()
with open('ALMANAC_drugcaption_dsv3.pkl', 'wb') as file:
    pickle.dump(data_dict, file)

# #deepseekr1
selected_columns = excel_data[['pubchemid', 'deepseekr1']]
data_dict = selected_columns.set_index('pubchemid')['deepseekr1'].to_dict()
with open('ALMANAC_drugcaption_dsr1.pkl', 'wb') as file:
    pickle.dump(data_dict, file)

#gemini
selected_columns = excel_data[['pubchemid', 'gemini']]
data_dict = selected_columns.set_index('pubchemid')['gemini'].to_dict()
with open('ALMANAC_drugcaption_ge.pkl', 'wb') as file:
    pickle.dump(data_dict, file)

#qwen
selected_columns = excel_data[['pubchemid', 'qwen']]
data_dict = selected_columns.set_index('pubchemid')['qwen'].to_dict()
with open('ALMANAC_drugcaption_qw.pkl', 'wb') as file:
    pickle.dump(data_dict, file)

#chatgpt
selected_columns = excel_data[['pubchemid', 'chatgpt']]
data_dict = selected_columns.set_index('pubchemid')['chatgpt'].to_dict()
with open('ALMANAC_drugcaption_gpt.pkl', 'wb') as file:
    pickle.dump(data_dict, file)


# # 加载
# with open('ALMANAC_drugcaption_dsv3.pkl', 'rb') as file:
#     data = pickle.load(file)
# print(data)

# with open('ALMANAC_drugcaption_dsr1.pkl', 'rb') as file1:
#     data1 = pickle.load(file1)
# print(data1)

# with open('ALMANAC_drugcaption_ge.pkl', 'rb') as file2:
#     data2 = pickle.load(file2)
# print(data2)

# with open('ALMANAC_drugcaption_qw.pkl', 'rb') as file3:
#     data3 = pickle.load(file3)
# print(data3)

# with open('ALMANAC_drugcaption_gpt.pkl', 'rb') as file4:
#     data4 = pickle.load(file4)
# print(data4)