import subprocess
import os
run_cmd = "python main.py --method GTA"
eval_cmd = "python eval.py --method GTA --model_best 1"

dataset_l = [
	"--dataselect 1 --dataroot digits --nepochs 50", 
	"--dataselect 2 --dataroot digits --nepochs 50", 
	"--dataselect 3 --dataroot images --nepochs 150", 
	"--dataselect 4 --dataroot images --nepochs 150",
	]
class_balance_l = ["--class_balance -1", "--class_balance 0.005"]
augmentation_l = ["--augmentation 0", "--augmentation 1"]
os.system("del experiment_log.txt")
for dataset in dataset_l:
	for class_balance in class_balance_l:
		for augmentation in augmentation_l:
			setting = ' '.join([dataset, class_balance, augmentation])
			curr_cmd = ' '.join([run_cmd, dataset, class_balance, augmentation])
			print(setting)
			with open('experiment_log.txt', 'a') as f:
				f.write(setting+'\n------------------\n\n')
			with open('experiment_log.txt', 'a') as f:
				subprocess.call(curr_cmd, stdout=f)
			curr_cmd = ' '.join([eval_cmd, dataset])
			with open('experiment_log.txt', 'a') as f:
				subprocess.call(curr_cmd, stdout=f)
			with open('experiment_log.txt', 'a') as f:
				f.write('\n\n------------------\n')