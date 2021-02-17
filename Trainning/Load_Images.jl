using Images, TestImages, Colors, FileIO
using Flux, CSV, DataFrames, Random
using Flux:onehotbatch, crossentropy, onecold, Data.DataLoader
using StatsBase:sample
using BSON: @save
#Incluimos la funciones necesarias para cargas los datos
include("/home/aquilesbailo/Traffic-Light-AI/Trainning/CNN_modelo.jl")
include("/home/aquilesbailo/Traffic-Light-AI/Trainning/CNN_functions.jl")
################################################################################
#Cargamos los datos a memoria
df_train = CSV.read("/home/aquilesbailo/Traffic-Light-AI/Data Bases/Circles/test_mode_circle_00.csv", DataFrame)
IND = convert(Array, df_train.Ind)
Y_rT = convert(Array, df_train.TRAIN)

#Tamaño del conjunto
m = length(Y_rT)
#Definimos las Clases
mClass = mClases(Y_rT)
#Iniciamos los Tensores de Datos
TI = Array{Float32}(undef, 50, 50, 1, m)
Y = onehotbatch(Y_rT, mClass);

#Definimos la ruta de los archivos
str_dir = "/home/aquilesbailo/Traffic-Light-AI/Data Bases/Circles/circle_00_test/"

#Creamos el tensor de imagenes de entrenamiento
for i in 1:m
    mImg = load(str_dir*string(i)*".jpg")
    TI[:,:,:,i] = mChannelImg(mImg)
end

################################################################################
#Arreglamos los datos de Entrenamiento

#Tamaño de la epoca
tN = 250
#Tamaño de la muestra
tM = 1000


function mTrain()
   mT = sample(1:m,tM,replace=false)
   data_TRAIN = TI[:,:,:,mT]
   Y_TRAIN = Y[:,mT]
   return DataLoader((data_TRAIN,Y_TRAIN),batchsize= tN)
end

################################################################################
#Entrenamos el modelo
#Inicio el tiempo de entrenamiento
t1 = time_ns()
mT = 200
opt = ADAM(0.001)
for i in 1:mT
   Flux.train!(Loss,params(modelo),mTrain(),opt);
   print(i, "\n")
end
#Fin tiempo de entrenamiento
t2 = time_ns()

#Guardamos los parámetros del modelo
SaveModel("circle05")

print("Tiempo Empleado: ", (t2-t1)/60.0e9, " minutos.")
