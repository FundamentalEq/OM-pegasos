import sys
fname = sys.argv[1]


total = []
for line in open(fname,'r') :
    line = line.strip()
    if '?' in line : continue
    line = line.split(',')
    for i in range(1,len(line)-1) :
        line[i] = str(chr(ord('a') + i - 1)) + ":" + line[i]
    line = ','.join(line)
    total.append(line)



split = int(len(total) * 0.8)

train = total[:split]
test = total[split:]

open(sys.argv[2],'w').write("\n".join(train))
open(sys.argv[3],'w').write("\n".join(test))