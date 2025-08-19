import translated_subtitler as tl
import time

filenames = ['testq.mp4','test2.mp4','test3.mp4'] #Put filenames here in list

start_time = time.time() #Program start time
for file in filenames: #For each file
    current_start_time = time.time() #Current file start time
    tl.translate_video(file) #Translate video
    current_end_time = time.time() #Current file end time
    current_total_seconds = current_end_time - current_start_time
    print('\n',f"{file} is done! Runtime: {current_total_seconds:.2f} seconds ({current_total_seconds/60:.2f} minutes)")
end_time = time.time() #end time
total_seconds = end_time - start_time
print('\n',f"Done! Total runtime: {total_seconds:.2f} seconds ({total_seconds/60:.2f} minutes)")