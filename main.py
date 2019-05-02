import sys
import mmhash
from sklearn.metrics import accuracy_score

class pypegasos:

    def __init__(self, nbits=18, lamb=.1):
        self.scaling = 1.
        self.nbits = nbits
        self.lamb = lamb
        self.biasKey = "BIAS"

        self.vectorSize = 2 ** nbits
        self.resetWeights()

    def resetWeights(self):
        self.weights = [0. for x in range(0, self.vectorSize)]

    def innerProduct(self, vec):
        """ Input format here is similar to Vowpal Wabbit:
            <feature_name>:<feature_value> """

        # Bias term always takes a constant value of 1 for all our observations
        # e.g., intercept in linear regression.
        innerProd = (self.weights[self._hash(self.biasKey)] *
                     self._rademacher(self.biasKey))

        for coord in vec:
            feature = coord.split(':')
            featureName = str(feature[0])
            featureVal = float(feature[1])
            
            wtVal = self.weights[self._hash(featureName)]
            radVal = self._rademacher(featureName)
            innerProd = innerProd + (featureVal * wtVal * radVal)

        return innerProd

    def processBatch(self, currentBatch, t):
        k = len(currentBatch)
        updateSet = []
        # Determine which items in batch contribute to loss
        for obs in currentBatch:
            if len(obs) < 2: continue
            y = int(obs[-1])
            if y==2: y = -1
            else: y = 1
            obs[-1] = y
            if y != -1 and y != 1:
                print("Disregarding observation because of invalid training \
                      observation.  Labels must be either -1 or 1")
                continue

            features = obs[1:-1]
            inner_product = self.innerProduct(features)

            if y * inner_product < 1:
                updateSet.append(obs)
        
        # Update weight vector using the update set
        self.updateWeights(self.lamb, t, k, updateSet)


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