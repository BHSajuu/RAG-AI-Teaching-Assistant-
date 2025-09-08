# converts the videos into audio format i.e .mp3
import os 
import subprocess

files = os.listdir('videos')
for file in files:
      tutorial_number = file.split(" [")[0].split(" #")[1].split(".")[0]
      file_name = file.split("_")[0]
      print(f"Processing tutorial {tutorial_number}: {file_name}")
      subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])