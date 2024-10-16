import subprocess
import tensorflow as tf

nvidia_smi_result = subprocess.run(
    ["nvidia-smi","-l"], 
    shell=True, 
    capture_output=True, 
    text=True,
)
print(nvidia_smi_result.stdout)
print(tf.config.list_physical_devices('GPU'))
