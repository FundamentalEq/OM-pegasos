import sys
import hashlib
from sklearn.metrics import accuracy_score
def H(val) :
    return hash(val)
nbits = 18

class pypegasos:

    def __init__(self):
        self.lamb = 0.1
        self.biasKey = "BABY KO BASE PASAND HA"
        self.vectorSize = 2 ** nbits
        self.resetWeights()

    def resetWeights(self):
        self.weights = []
        self.weights = [0.0] * self.vectorSize

    def getVal(self,coord) :
        feature = coord.split(':')
        featureName = str(feature[0])
        featureVal = float(feature[1])
        return featureName, featureVal
    def innerProduct(self, vec):
        innerProd = 0
        for i,coord in enumerate(vec):
            featureName, featureVal = self.getVal(coord)
            wtVal = self.weights[self._hash(featureName)]
            radVal = -1
            innerProd += (featureVal * wtVal * radVal)
        return innerProd

    def processBatch(self, currentBatch, t):
        k = len(currentBatch)
        updateSet = []
        # Determine which items in batch contribute to loss
        for obs in currentBatch:
            if len(obs) < 2: 
                continue
            y = int(obs[-1])
            if y==2 : y = -1
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
    def updateWeights(self, lamb, t, k, updateSet):
        stepKeys = self.gradientStep(lamb, t, k, updateSet)
        self.projectStep(lamb, stepKeys)

    def gradientStep(self, lamb, t, k, updateSet):
        eta = 1./(lamb * t)
        stepDirection = {}

        # Initialize stepDirection with key corresponding to bias term

        stepDirection[self.biasKey] = 0
        # Compute the subgradient of our loss function at the present update set
        for obs in updateSet:
            if len(obs) < 2: continue

            y = int(obs[-1])
            assert (y==-1 or y == 1)
            # Compute subgradient of bias(intercept) term
            stepDirection[self.biasKey] += y

            # Compute subgradient of features
            features = obs[1:-1]
            for coord in features:
                featureName, featureVal = self.getVal(coord)
                featureVal *= y
                if featureName not in stepDirection:
                    stepDirection[featureName] = 0.0
                stepDirection[featureName] += featureVal

        scaling = eta/k

        # Update weight coefficients
        for i,(key,val) in enumerate(stepDirection.items()):
            indx = self._hash(key)
            rad = -1

            self.weights[indx] *= (1 - (eta * lamb))
            self.weights[indx] += scaling * val * rad
        
        return stepDirection.keys()

    def projectStep(self, lamb, stepKeys):

        normSquared = 0.

        for feature in stepKeys:
            indx = self._hash(feature)
            normSquared += self.weights[indx] ** 2

        scaling = 1
        if normSquared != 0:
            scaling = min(1, (1/(lamb ** .5))/(normSquared ** .5))

        if scaling != 1:
            for feature in stepKeys:
                indx = self._hash(feature)
                self.weights[indx] *= scaling
            

    def _hash(self, val):
        return H(val) % self.vectorSize

    def _norm(self):
        sqNorm = sum(map(lambda x: x*x, self.weights))
        return sqNorm ** .5

vector = pypegasos()
k = sys.argv[3]
batchSize = int(k) # Batch size. k=1 is SGD, k=N is batch GD
t = 1
filename = sys.argv[1]
test = sys.argv[2]
currentBatch = []
inputfile = open(filename, 'r')
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
    yi = obs[-1]
    if yi == 4: y = 1
    else: y = -1
    y_true.append(y)
    instance = obs[1:-1]
    prediction = vector.innerProduct(instance)
    if prediction > 0: p = 1
    else: p =-1
    y_pred.append(p)

print(accuracy_score(y_true,y_pred))