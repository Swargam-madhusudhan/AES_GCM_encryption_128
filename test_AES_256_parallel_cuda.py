import subprocess
import re

if "check_output" not in dir( subprocess ):
    def f(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)
        return output
    subprocess.check_output = f
	
nCorrect = 0

for i in range(21):
	datasetDir = "./Dataset_256/" + str(i) + "/"
	result = subprocess.check_output(["./build/AES_Encrypt_GCM_256_Parallel_CUDA","-i",datasetDir+"PT.dat","-e",datasetDir+"CT.dat","-t","vector"])
	correct = re.search(b'"correctq": true',result) != None
	if correct:
		nCorrect += 1
	else:
		print(str(i) + " incorrect")
	ComputeMatch = re.search(b'\\{.*"elapsed_time": (\\d+).*"message": "Performing CUDA computation"',result)
	ExecuteTime = int(ComputeMatch.group(1))
	lenMatch = re.search(b'The input length is (\d+)', result)
	input_size_bytes = int(lenMatch.group(1))
	execution_time_seconds = ExecuteTime / 1e9
	throughput_gbs = (input_size_bytes / 1e9) / execution_time_seconds
	print("Dataset/" + str(i) + ": Input Size: " + str(input_size_bytes) + " bytes" + ": Execute Time:" + str(ExecuteTime) + ": Throughput: " + str(throughput_gbs) + " GB/s" )

print(str(nCorrect) + " / 21 correct")
