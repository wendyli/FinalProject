import os

for filename in os.listdir("."):
	if filename.startswith("train"):
		new_name = filename[len('train'):]
		new_name = new_name[: 0- len('.jpg')]
		num = 40100 + int(new_name)
		os.rename(filename, 'train_{}.jpg".format(num))
