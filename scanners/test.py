from dlisio import dlis
from mappings.WellLogsFormat import WellLogFormat
from worker.tasks import convert_to_json_task
from datetime import datetime
import os
from utils.logger import Logger

#pending work to be done before release
# change the checksum logic to calculate on output files - done
# change the print to logger - done
# add traceback to exceptions - done
# add for csv updates in task.py for failed

# file_full_path = rf'F:\PyCharmProjects\mivaa-las-dlis-to-json-convertor\samples\11_30a-_9Z_dwl_DWL_WIRE_238615014.las'
#
# result = convert_to_json_task(filepath=file_full_path,
#                               output_folder="F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed",
#                               file_format=WellLogFormat.LAS.value)
# print(f"Task submitted for las file {file_full_path}, Task ID: {result}")
# file_full_path = rf'F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\uploads\2_1-A-14_B__WELL_LOG__WL_GR-DEN-NEU_MWD_5.DLIS'
# # log_filename = f'{os.path.basename(str(file_full_path))}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
# # file_logger = Logger.get_logger(log_filename)
# # file_logger.info(f"New file detected: {file_full_path}")
#
# print(f"Identified as DLIS: {file_full_path}. Extracting logical files for scanning")
# logical_files = dlis.load(file_full_path)
# print(f"Loaded {len(logical_files)} logical files from DLIS {file_full_path}")
#
# for logical_file in logical_files:
#     # print(str(logical_file.fileheader.id))
#     #how extactly does this github copilot work ? can i cha
#     # print(None)
#     try:
#         logical_file_id = str(logical_file.fileheader.id)
#     except Exception as e:
#         print(f"Error accessing logical file header in {file_full_path}: {e}")
#         continue  # Skip this logical file but continue processing others
#
#     print(f'this executed for {logical_file_id}')
#     result = convert_to_json_task(filepath=file_full_path,
#                                   output_folder="F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed",
#                                   file_format=WellLogFormat.DLIS.value,
#                                   logical_file_id=str(logical_file.fileheader.id))
#     print(f"Task submitted for logical file {logical_file.fileheader.id} in DLIS file {file_full_path}, Task ID: {result}")

# import pandas as pd
# import tempfile
# import os
# from langchain_community.document_loaders.csv_loader import CSVLoader
#
# # Step 1: Load CSV into a DataFrame
# file_path = r'F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\DLIS_DLISParametersProcessor_summary.csv'
# df = pd.read_csv(file_path)
#
# #remove all the columns from dataframe that has all the values as null
# df = df.dropna(axis=1, how='all')
#
# # Step 3: Save processed DataFrame to a temporary CSV file
# with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
#     temp_file_path = temp_csv.name  # Get the temp file path
#     df.to_csv(temp_file_path, index=False)
#
# # Step 4: Load the processed CSV using CSVLoader
# loader = CSVLoader(file_path=temp_file_path)
# data = loader.load()
#
# # Step 5: Delete the temporary CSV file after loading
# os.remove(temp_file_path)
#


# #working on prompts
# from langchain_core.prompts import ChatPromptTemplate
# #trying to instantiate an llm
# from langchain_openai import AzureChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
#
# map_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Your name is mivaa. You need to act like a oil and gas {user_persona} expert. Who has an expertise in well log interpretation and analysis."),
#         ("human", "Can you rag following details {user_input} from the {section} of the well log file and also give some insight from the {user_persona} perspective?"),
#     ]
# )



#
# map_chain = map_prompt_template | llm | StrOutputParser()
#
# # Define the values for the template
# variables = {
#     "user_persona": "petrophysicist",
#     "user_input": data[0].page_content,
#     "section": "parameter"
# }
#
# # Invoke the chain with variables
# response = map_chain.invoke(variables)
#
# print(response)