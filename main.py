import sys
import mmhash
from sklearn.metrics import accuracy_score

vector = pypegasos(nbits)
k = sys.argv[3]
batchSize = k # Batch size. k=1 is SGD, k=N is batch GD
t = 1
lamb = lamb
filename = sys.argv[1]
# dev = sys.argv[2]
test = sys.argv[2]
currentBatch = []
inputfile = open(filename, 'r')
devfile = open(dev,'r')
testfile = open(test,'r')
for line in inputfile:
    splitLine = line.split(',')
    currentBatch.append(splitLine)

    if len(currentBatch) >= batchSize:
        vector.processBatch(currentBatch, t)
        t = t + 1
        currentBatch = []

        if len(currentBatch) > 0:
            vector.processBatch(currentBatch, t)

inputfile.close()
y_true = []
y_pred = []
for line in testfile:
    splitLine = line.split(',')
    obs = splitLine
    y_true.append(obs[-1])
    instance = obs[1:-1]
    prediction = vector.innerProduct(instance)
    y_pred.append(prediction)

print(accuracy_score(y_true,y_pred))