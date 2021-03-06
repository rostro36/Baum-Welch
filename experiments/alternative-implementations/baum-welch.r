#source:
# https://github.com/adeveloperdiary/HiddenMarkovModel


forward = function(v, a, b, initial_distribution){
  
  T = length(v)
  M = nrow(a)
  alpha = matrix(0, T, M)
  
  alpha[1, ] = initial_distribution*b[, v[1]]
  
  for(t in 2:T){
    tmp = alpha[t-1, ] %*% a
    alpha[t, ] = tmp * b[, v[t]]
  }
  return(alpha)
}
 
backward = function(v, a, b){
  T = length(v)
  M = nrow(a)
  beta = matrix(1, T, M)
  
  for(t in (T-1):1){
    tmp = as.matrix(beta[t+1, ] * b[, v[t+1]])
    beta[t, ] = t(a %*% tmp)
  }
  return(beta)
}
 
 
BaumWelch = function(v, a, b, initial_distribution, n.iter = 100){
 
  for(i in 1:n.iter){
    print(i)
    T = length(v)
    M = nrow(a)
    K=ncol(b)
    alpha = forward(v, a, b, initial_distribution)
    beta = backward(v, a, b)
    xi = array(0, dim=c(M, M, T-1))
    
    for(t in 1:T-1){
      denominator = ((alpha[t,] %*% a) * b[,v[t+1]]) %*% matrix(beta[t+1,]) 
      for(s in 1:M){
        numerator = alpha[t,s] * a[s,] * b[,v[t+1]] * beta[t+1,]
        xi[s,,t]=numerator/as.vector(denominator)
      }
    }
    
    
    xi.all.t = rowSums(xi, dims = 2)
    a = xi.all.t/rowSums(xi.all.t)
    
    gamma = apply(xi, c(1, 3), sum)  
    gamma = cbind(gamma, colSums(xi[, , T-1]))
    for(l in 1:K){
      b[, l] = rowSums(gamma[, which(v==l)])
    }
    b = b/rowSums(b)
    
    #ADDED AFTERWARDS. NOT ORIGINAL IMPLEMENTATION
    initial_distribution=gamma[,1]
      
  }
  return(list(a = a, b = b, initial_distribution = initial_distribution))
}
 
data = read.csv("../test_matrices/observations.csv",header = FALSE)+1
data = as.matrix(unname(data))

emissionMatrix<- read.csv(file = '../test_matrices/emissionMatrix.csv',header=FALSE)
emissionMatrix<-as.matrix(unname(emissionMatrix))

transitionMatrix<- read.csv(file = '../test_matrices/transitionMatrix.csv',header=FALSE)
transitionMatrix<-as.matrix(unname(transitionMatrix))

stateProb<- read.csv(file = '../test_matrices/stateProb.csv',header=FALSE)
stateProb <-as.vector(t(unname(stateProb)))



#data = read.csv("../wikipedia_matrices/observations.csv",header = FALSE)+1
#data = as.matrix(unname(data))

#emissionMatrix<- read.csv(file = '../wikipedia_matrices/emissionMatrix.csv',header=FALSE)
#emissionMatrix<-as.matrix(unname(emissionMatrix))

#transitionMatrix<- read.csv(file = '../wikipedia_matrices/transitionMatrix.csv',header=FALSE)
#transitionMatrix<-as.matrix(unname(transitionMatrix))

#stateProb<- read.csv(file = '../wikipedia_matrices/piMatrix.csv',header=FALSE)
#stateProb <-as.vector(t(unname(stateProb)))

#data = read.csv("data_r.csv")

M=2
K=3


A = transitionMatrix
B = emissionMatrix
initial_distribution = stateProb


b = B
a = A
v = data


(myout = BaumWelch(data, A, B, initial_distribution, n.iter = 1))
