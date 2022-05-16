################################################################################
# El código está implementado a calcular el coeficiente de precisión por clase
# Además representa los coeficientes por medio de un histograma y heatmap
################################################################################
using DataFrames, CSV, Plots
include("/home/aquilesbailo/Traffic-Light-AI/Trainning/CNN_functions.jl")
################################################################################
#Definimos el nombre del modelo a evaluar
name = "circle08_P1"
str_res = "/home/aquilesbailo/Traffic-Light-AI/Trainning/results/circle_06/"
################################################################################

#Cargamos los parametros del modelo requerido
LoadModel(name)
Flux.testmode!(modelo[13], true)

#Obtenemos el vector de predicción. Esta es una versión no mutable de onecold(modelo(TI))
y_FINAL = Array{Float64}(undef,m)   
for i in 1:m
    Zipi = modelo(TI[:,:,:,i:i])
    y_FINAL[i,:] = onecold(Zipi, mClass)
end

################################################################################
#Salvamos el resultado en un archivo csv
T_result = DataFrame(Ind = IND, RESULT = y_FINAL)
CSV.write(str_res*name*".csv", T_result);

################################################################################
#Obtenemos el vector de valoración cruzada y los coeficientes por clases
Pred = (y_FINAL .== Y_rT)

mBJ = reshape(Pred, 1000, :)'
mPC = zeros(10, 10)
mPB = zeros(11)

for i in 1:10
    mPB[i] = mCoef(mBJ[i, :])
    mTC = reshape(mBJ[i, :], 100, :)'
    for j in 1:10
        mPC[i,j] = mCoef(mTC[j, :])
    end
end
mPB[end] = mCoef(Pred)

################################################################################
#Creamos los graficos apropiados
clases = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Total"]
p1 = bar(clases, mPB,
    xticks= (0:0.05:1),
    orientation =:h,
    fillcolor=[:blue,:green,:pink,:red,:gray, :purple],
    fillalpha = 0.6,
    label = "",
    framestyle =:box,
    title = ""
    )

x_pr = convert(Array,1:1:10)
y_pr = convert(Array,1:1:10)
p2 = heatmap(x_pr, y_pr, mPC',
    framestyle =:box,
    xticks= (1:10, [ "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]),
    yticks= ((1:1:10)),
    title = "",
    size =(700, 500)
     )
plot(p2, p1, size =(1700, 500))
png(str_res*name)
