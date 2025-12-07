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


for i in range(20):
	datasetDir = "./Dataset/" + str(i) + "/"
	result = subprocess.check_output(["./build/AES_Encrypt_GCM_128_Seq","-i",datasetDir+"PT.dat","-e",datasetDir+"CT.dat","-t","vector"])
	correct = re.search(b'"correctq": true',result) != None
	if correct:
		nCorrect += 1
	else:
		print(str(i) + " incorrect")
	ComputeMatch = re.search(b'\\{.*"elapsed_time": (\\d+).*"message": "Performing CUDA computation"',result)
	ExecuteTime = int(ComputeMatch.group(1))
	print("Dataset/" + str(i) + "Execute Time:" + str(ExecuteTime))


print(str(nCorrect) + " / 20 correct")
