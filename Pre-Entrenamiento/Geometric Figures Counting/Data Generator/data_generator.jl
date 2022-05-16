using Images
using DataFrames, CSV

include("/home/aquilesbailo/Traffic-Light-AI/data_Functions.jl")

#Definimos la ruta de los archivos de entrenamiento
str_train = "/home/aquilesbailo/Traffic-Light-AI/Data Bases/Circles/circle_06/"

IND = zeros(11000)
train = zeros(11000)

for i in 0:10999

    #Obtenemos la clase de la imagen
    p = convert(Int64, floor(i/1000))

    #Obtenemos el valor de la clase de manera aleatoria (para testmode)
    #p = convert(Int64, floor(10*rand()))

    #Creamos la imagen
    img  = mCircles(50,50,p)
    save(str_train*string(i+1)*".jpg",img)

    #Salvamos la clase de cad imagen
    IND[i+1] = i+1
    train[i+1] = p
end

#Salvamos los indices de las imagenes
Table = DataFrame(Ind = IND, TRAIN = train)
CSV.write("train_mode_circle_06.csv", Table);
