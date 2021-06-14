

require(here)
require(psych)
require(LNIRT)
require(MASS)

options(max.print = 1000000)

################################################################################

# Import Form 1 data

form1 <- read.csv('data/Form1.csv')

# Include only dichotomous item responses and response times

d <- form1[,211:610]
describe(d)

# Make everything numeric

for(i in 1:ncol(d)){
  d[,i] <- as.numeric(d[,i])
}

# If response time is 0, make it 1, then take the log of RT

for(i in 201:ncol(d)){
  if(length(d[,i]==0)>0){
    d[which(d[,i]==0),i] = 1
  }
  
  d[,i] = log(d[,i])
}




describe(d)


###############################################################################

# Fit Response Time model through LNIRT package

fit <- LNRT(RT     = d[,201:400],
             XG     = 5000,
             burnin = 20,
             td     = FALSE,
             WL     = TRUE)


tau   <- as.numeric(fit$Post.Means$Person.Speed)

lambda <- as.numeric(fit$Post.Means$Sigma2)

beta <- as.numeric(fit$Post.Means$Time.Intensity)

Sigma <- matrix(c(fit$Post.Means$Var.Person.Ability,
                  fit$Post.Means$Cov.Person.Ability.Speed,
                  fit$Post.Means$Cov.Person.Ability.Speed,
                  fit$Post.Means$Var.Person.Speed),2,2)

describe(tau)
describe(lambda)
describe(beta)


# Fit 2PL through mirt package

require(mirt)

twopl <- mirt(data = d[,1:200],
              model = 1,
              itemtype = 'Rasch')

b <- coef(twopl,IRTpars=TRUE,simplify=TRUE)$items[,2]

th <- fscores(twopl,method='ML')[,1]

describe(th)
describe(b)


Sigma <- cov(cbind(th,tau))

cov2cor(Sigma)

################################################################################
# Function to generate an item response given the theta parameter and item parameters

gen.iresp <- function(theta,a,b){
  
  const <- a*(theta-b)
  p     <- exp(const)/(1+exp(const))
  
  (p>runif(1,0,1))*1
  
}


# Function to generate item response time given tau and item parameters

gen.idur <- function(tau,lambda,beta){
  
  mu <- beta - tau
  sd <- 1/lambda
  
  rnorm(1,mu,sd)
  
}

################################################################################
set.seed(06142021)

N = 1000000

resp <- matrix(nrow=N,ncol=200)
rt   <- matrix(nrow=N,ncol=200)

for(i in 1:N){
  
  true.theta.tau <- mvrnorm(1,mu=c(0,0),Sigma = Sigma)
  
  theta <- true.theta.tau[1]
  tau   <- true.theta.tau[2]
  
  for(j in 1:200){
    
    resp[i,j] = gen.iresp(theta = theta,
                          a=1,
                          b=b[j])
    
    rt[i,j]   = gen.idur(tau = tau,
                         lambda = lambda[j],
                         beta = beta[j])
    
  }
  
}


data1 <- data.frame(cbind(resp,rt))
data1$target <- 0

################################################################################

resp2 <- matrix(nrow=N,ncol=200)
rt2   <- matrix(nrow=N,ncol=200)

for(i in 1:N){
  
  true.theta.tau <- mvrnorm(1,mu=c(0,0),Sigma = Sigma)
  
  theta <- true.theta.tau[1]
  tau   <- true.theta.tau[2]
  
  # Item preknowledge effect on latent ability
  # Randomly draw a number from a uniform distribution, U[0,1]
  # this improves the odds of getting the item correct by exp(delta)
  # delta is the change in ability
  
  theta2 <- theta + runif(1,0,1)
  
  
  # Item preknowledge effect on latent speed
  # Randomly draw a number from a uniform distribution, U[.2,1.5]
  # this reduces the response time by
  # 1 - [exp(4-tau - delta)/exp(4 - tau)]
  
  tau2 <- tau + runif(1,.2,1.5)
  
  
  # Randomly draw a number between 0.1 and 1
  # This is the proportion of compromised items the individual had accessed
  
  pci <- runif(1,0.1,1)
  nci <- round(200*pci )

  ci       <- sort(sample(1:200,nci))
  ci.prime <- setdiff(1:200,ci) 
  
  for(j in ci.prime){
    
    resp2[i,j] = gen.iresp(theta = theta,
                           a=1,
                           b=b[j])
    
    rt2[i,j]   = gen.idur(tau = tau,
                          lambda = lambda[j],
                          beta = beta[j])
    
  }
  
  for(j in ci){
    
    resp2[i,j] = gen.iresp(theta = theta2,
                           a=1,
                           b=b[j])
    
    rt2[i,j]   = gen.idur(tau = tau2,
                          lambda = lambda[j],
                          beta = beta[j])
    
  }
  
}


data2 <- data.frame(cbind(resp2,rt2))
data2$target <- 1
################################################################################

j = 52

describe(d[,j])
describe(data1[,j])
describe(data2[,j])


describe(d[,j+200])
describe(data1[,j+200])
describe(data2[,j+200])



plot(describe(d[,1:200])$mean,describe(data1[,1:200])$mean)
abline(0,1)
plot(describe(d[,1:200])$mean,describe(data2[,1:200])$mean)
abline(0,1)

plot(describe(d[,201:400])$mean,describe(data1[,201:400])$mean)
abline(0,1)

plot(describe(d[,201:400])$mean,describe(data2[,201:400])$mean)
abline(0,1)


################################################################################

train <- rbind(data1,data2)

rm(resp,resp2,rt,rt2,Sigma,b,beta,ci,ci.prime,i,j,lambda,N,nci,pci,tau,tau2,
   th,theta,theta2,true.theta.tau,gen.idur,gen.iresp)

save.image("B:/Ongoing_Research/XGBOOST/MS2/xgboost2/data/simulated data.RData")

################################################################################

require(xgboost)
require(pROC)

# Add average response time and average response accuracy as features

train$ave.rt <- rowMeans(train[,201:400])
train$ave.r  <- rowMeans(train[,1:200])


# Split the data into training and test (80-20 split)

loc <- sample(1:nrow(train),nrow(train)*.20)

df_train <- train[-loc,]
df_test  <- train[loc,]

# Create the xgb.DMatrix objects 

dtrain <- xgb.DMatrix(data = data.matrix(df_train[,-401]), label=df_train[,401])
dtest  <- xgb.DMatrix(data = data.matrix(df_test[,-401]),  label=df_test[,401])

# Fit the XGBoost model with the tuned parameters

# Tuning the parameters is a whole different story, and I tried to explain it in the paper.
# Maybe, I can do another post about it, but there is already a lot of resources on the web about it.

watchlist <- list(train=dtrain, test=dtest)

bst <- xgb.train(data              = dtrain,
                 nround            = 10000,
                 eta               = .05,
                 min_child_weight  = 1,
                 max_depth         = 5,
                 gamma             = 0,
                 max_delta_step    = 0,
                 subsample         = 1,
                 colsample_bytree  = 1,
                 lambda            = 0,
                 alpha             = 0,
                 scale_pos_weight  = 1,
                 num_parallel_tree = 1,
                 nthread           = 15, 
                 objective         = 'binary:logistic',
                 eval_metric       = 'rmse',
                 watchlist         = watchlist, 
                 early_stopping_round  = 100)

# Predict the outcome for the test dataset based on the model 

df_test$prob <- predict(bst,dtest)

auc(df_test$target,df_test$prob)


th = quantile(df_test[df_test$target==0,]$prob,c(.95,.99))
th


# Confusion matrix corresponding to .05 Type I error rate

table(df_test$target,df_test$prob>th[1])

# Confusion matrix corresponding to .01 Type I error rate

table(df_test$target,df_test$prob>th[2])


################################################################################

df_form1 <- d
colnames(d) <- colnames(train[,1:400])
df_form1$ave.rt <- rowMeans(df_form1[,201:400],na.rm=TRUE)
df_form1$ave.r  <- rowMeans(df_form1[,1:200],na.rm=TRUE)


dform <- xgb.DMatrix(data = data.matrix(df_form1))   #, label=form1$Flagged)

pred <- predict(bst,dform)


auc(form1$Flagged,pred)


th = quantile(pred[form1$Flagged==0],c(.95,.99))
th


# Confusion matrix corresponding to .05 Type I error rate

table(form1$Flagged,pred>th[1])

# Confusion matrix corresponding to .01 Type I error rate

table(form1$Flagged,pred>th[2])


table(form1$Flagged,pred>0.95)













