# Redes Neuronales Artificiales

#importando dataset

dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[,4:14]

# Codificar datos categoricos
dataset$Geography = as.numeric(factor(dataset$Geography,
                         levels = c("France","Spain","Germany"),
                         labels = c(1,2,3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                           levels = c("Female","Male"),
                           labels = c(0,1)))

#Dividir data set entre conjunto de entranamiento y testing
library(caTools)
set.seed(123) #Seleccionando semilla
split = sample.split(dataset$Exited,SplitRatio = 0.8)
#Separando datos de entrenamiento y testing
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

#Escalado de valores
training_set[,1:10] = scale(training_set[,1:10])
testing_set[,1:10] = scale(testing_set[,1:10])

#Crear la red neuronal
#install.packages("h2o")
library(h2o)
h2o.init(nthreads = -1)
#hidden c(6,6)Primera capa con 6, segunda capa con 6, etc..
classifier = h2o.deeplearning(y = "Exited",
                              training_frame = as.h2o(training_set),
                              activation = "Rectifier",
                              hidden = c(6,6),
                              epochs = 150,
                              train_samples_per_iteration = -2)

# Predicción de los resultados con el conjunto de testing
prob_pred = h2o.predict(object = classifier,newdata = as.h2o(testing_set[,-11]))
y_pred = prob_pred> 0.5
y_pred = as.vector(y_pred)

# Crear la matriz de confusión
cm = table(testing_set[, 11], y_pred)

#Cerrar la sesion de H2O
h2o.shutdown()
