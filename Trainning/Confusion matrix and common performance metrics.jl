using CSV, DataFrames, Random
using DelimitedFiles
using DataFrames

#Cargamos los datos a memoria
df_train = CSV.read("/home/aquilesbailo/Traffic-Light-AI/Trainning/results/circle_06_test/circle08.csv", DataFrame)
Y_P = convert(Array{Int64,1}, df_train.RESULT)

#creamos la matriz
A = Array{Int64}(undef, 11, 11)

for i in 1:11
    lim_A = (i-1)*(1000) + 1
    lim_B = i*(1000)

    for j in 0:10
        A[j+1, i] = (count(k -> k == j, Y_P[lim_A:lim_B]))
    end
end
T_result = DataFrame(A)

display(T_result)
