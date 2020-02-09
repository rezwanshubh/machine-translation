file1 = open('myfile.txt', 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
for line in Lines:
    if (count<2):
        file2 = open('myfile_2.txt', 'w')
        file2.write(line)
        count = count + 1