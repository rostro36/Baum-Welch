
emissionMatrix<- read.csv(file = 'init/emissionMatrix.csv',header=FALSE)
emissionMatrix<-as.matrix(unname(emissionMatrix))

transitionMatrix<- read.csv(file = 'init/transitionMatrix.csv',header=FALSE)
transitionMatrix<-as.matrix(unname(transitionMatrix))

#add one because R encodes states from 1 to n (not 0 to n-1)
observations<- read.csv(file = 'init/observations.csv',header=FALSE)
observations<-t(observations)+1

stateProb<- read.csv(file = 'init/stateProb.csv',header=FALSE)
stateProb <-t(stateProb)

steps<- read.csv(file = 'result/steps.csv',header=FALSE)
#steps <-steps[1]


#emissionMatrix<- read.csv(file = 'wikipedia_matrices/emissionMatrix.csv',header=FALSE)
#emissionMatrix<-as.matrix(unname(emissionMatrix))

#transitionMatrix<- read.csv(file = 'wikipedia_matrices/transitionMatrix.csv',header=FALSE)
#transitionMatrix<-as.matrix(unname(transitionMatrix))

#add one because R encodes states from 1 to n (not 0 to n-1)
#observations<- read.csv(file = 'wikipedia_matrices/observations.csv',header=FALSE)
#observations<-t(observations)+1

#stateProb<- read.csv(file = 'wikipedia_matrices/piMatrix.csv',header=FALSE)
#stateProb <-t(stateProb)


hmm = initHMM(seq(1,length(stateProb), by=1),seq(1,dim(emissionMatrix)[2], by=1),startProbs=stateProb, transProbs=transitionMatrix, emissionProbs=emissionMatrix)

print(hmm)
# Baum-Welch
bw = baumWelch(hmm,observations,maxIterations = steps[1,1])
print(bw$hmm)


result_emissionMatrix<- read.csv(file = 'result/emissionMatrix.csv',header=FALSE)
result_emissionMatrix<-as.matrix(unname(result_emissionMatrix))

result_transitionMatrix<- read.csv(file = 'result/transitionMatrix.csv',header=FALSE)
result_transitionMatrix<-as.matrix(unname(result_transitionMatrix))

#add one because R encodes states from 1 to n (not 0 to n-1)
result_observations<- read.csv(file = 'result/observations.csv',header=FALSE)
result_observations<-t(result_observations)+1

result_stateProb<- read.csv(file = 'result/stateProb.csv',header=FALSE)
result_stateProb <-t(result_stateProb)

print(norm(result_emissionMatrix - bw$hmm$emissionProbs))
print(norm(result_transitionMatrix - bw$hmm$transProbs))
print(norm(result_stateProb- bw$hmm$startProbs))


