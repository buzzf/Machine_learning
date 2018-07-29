with open('az.txt','w') as f:
	for i in range(65, 91):
		f.write(chr(i))
		f.write('\n')
	for j in range(97,122):
		f.write(chr(j))
		f.write('\n')