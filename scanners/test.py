from dlisio import dlis
from mappings.WellLogsFormat import WellLogFormat
from worker.tasks import convert_to_json_task
from datetime import datetime
import os
from utils.logger import Logger
#
# #pending work to be done before release
# # change the checksum logic to calculate on output files - done
# # change the print to logger - done
# # add traceback to exceptions - done
# add for csv updates in task.py for failed
#
# file_full_path = rf'F:\PyCharmProjects\mivaa-las-dlis-to-json-convertor\samples\11_30a-_9Z_dwl_DWL_WIRE_238615014.las'
#
# result = convert_to_json_task(filepath=file_full_path,
#                               output_folder="F:\PyCharmProjects\mivaa-las-dlis-to-json-convertor\processed",
#                               scanner_cls=LasScanner,
#                               file_format=WellLogFormat.LAS)
# print(f"Task submitted for las file {file_full_path}, Task ID: {result}")
file_full_path = rf'F:\PyCharmProjects\mivaa-las-dlis-to-json-convertor\uploads\2_1-A-14_B__WELL_LOG__WL_GR-DEN-NEU_MWD_5.DLIS'
# log_filename = f'{os.path.basename(str(file_full_path))}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
# file_logger = Logger.get_logger(log_filename)
# file_logger.info(f"New file detected: {file_full_path}")

print(f"Identified as DLIS: {file_full_path}. Extracting logical files for scanning")
logical_files = dlis.load(file_full_path)
print(f"Loaded {len(logical_files)} logical files from DLIS {file_full_path}")

for logical_file in logical_files:
    # print(str(logical_file.fileheader.id))
    #how extactly does this github copilot work ? can i cha
    # print(None)
    try:
        logical_file_id = str(logical_file.fileheader.id)
    except Exception as e:
        print(f"Error accessing logical file header in {file_full_path}: {e}")
        continue  # Skip this logical file but continue processing others

    print(f'this executed for {logical_file_id}')
    result = convert_to_json_task(filepath=file_full_path,
                                  output_folder="F:\PyCharmProjects\mivaa-las-dlis-to-json-convertor\processed",
                                  file_format=WellLogFormat.DLIS.value,
                                  logical_file_id=str(logical_file.fileheader.id))
    print(f"Task submitted for logical file {logical_file.fileheader.id} in DLIS file {file_full_path}, Task ID: {result}")
#
#
# # import shutil
# # import os
# #
# # # Define source and destination folders
# # source_folder = rf"F:\logs-scanner-directory\uploads"  # Change this to your source folder
# # destination_folder = rf"F:\PyCharmProjects\mivaa-las-dlis-to-json-convertor\uploads"  # Change this to your destination folder
# #
# # # List of files to copy
# # files_to_copy = [
# #     ".azDownload-72597339-e003-ee47-4c61-9a5b191dce38-WL_RAW_PROD_CAL-CCL-ELEM-FLOW-GR_2015-10-27_14.DLIS",
# #     "1_3-10__WELL_LOG__WL_RAW_GR-REMP_MWD_1.DLIS",
# #     "1_3-12_S__WELL_LOG__WL_RAW_GR-REMP_MWD_1.DLIS",
# #     "1_3-7_T2__PETROPHYSICS__L0804CMP.DLIS",
# #     "1_3-7_T3__PETROPHYSICS__L0804CMP.DLIS",
# #     "1_3-7_T3__PETROPHYSICS__L0804CPI.DLIS",
# #     "1_3-7_T3__PETROPHYSICS__L0804CPI__02.DLIS",
# #     "1_3-7_T3__ROCK_AND_CORE__L0804CPI.DLIS",
# #     "1_3-7_T3__WELL_LOG__L0804CMP.DLIS",
# #     "1_3-7_T3__WELL_PATH__L0804CPI.DLIS",
# #     "1_3-7_T3__WELL_SEISMIC__L0804CPI.DLIS",
# #     "1_3-7__PETROPHYSICS__L0804CMP.DLIS",
# #     "1_3-7__PETROPHYSICS__L0804CPI.DLIS",
# #     "1_3-7__PETROPHYSICS__L0804CPI__02.DLIS",
# #     "1_3-7__ROCK_AND_CORE__L0804CPI.DLIS",
# #     "1_3-7__WELL_LOG__L0804CMP.DLIS",
# #     "1_3-7__WELL_PATH__L0804CPI.DLIS",
# #     "1_3-7__WELL_SEISMIC__L0804CPI.DLIS",
# #     "1_3-K-5_A__WELL_LOG__WLC_RAW_DEN-GR-NEU-REMP-SON_MWD_1.DLIS",
# #     "1_3-K-5_A__WELL_LOG__WL_RAW_DEN-GR-NEU-REMP_MWD_1 - Copy - Copy.DLIS",
# #     "1_3-K-5_T2__WELL_LOG__WL_RAW_SON_MWD_5.DLIS",
# #     "watcher.log"
# # ]
# #
# # # Ensure the destination folder exists
# # os.makedirs(destination_folder, exist_ok=True)
# #
# # # Copy each file
# # for file in files_to_copy:
# #     source_file = os.path.join(source_folder, file)
# #     destination_file = os.path.join(destination_folder, file)
# #
# #     if os.path.exists(source_file):
# #         shutil.copy2(source_file, destination_file)  # Preserve metadata
# #         print(f"Copied: {file}")
# #     else:
# #         print(f"File not found: {file}")
# #
# # print("File copying process completed.")
#
#
# import os
# import shutil
#
# # Define folders
# source_folder = rf"F:\PyCharmProjects\mivaa-las-dlis-to-json-convertor\processed"  # Folder to get the file list
# target_folder = rf"F:\logs-scanner-directory\processed"  # Folder to find matching files
# destination_folder = rf"F:\logs-scanner-directory\test"  # Folder to copy the matched files
#
# # Ensure destination folder exists
# os.makedirs(destination_folder, exist_ok=True)
#
# # Get the list of filenames from the source folder
# file_names = set(os.listdir(source_folder))  # Use set for faster lookup
#
# # Copy matching files from target_folder to destination_folder
# for file_name in file_names:
#     source_path = os.path.join(target_folder, file_name)
#     dest_path = os.path.join(destination_folder, file_name)
#
#     # Check if the file exists in the target folder
#     if os.path.isfile(source_path):
#         shutil.copy2(source_path, dest_path)
#         print(f"Copied: {file_name}")
#     else:
#         print(f"Not found: {file_name}")
#
# print("File copying process completed!")