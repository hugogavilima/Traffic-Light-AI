################################################################################
# El código está implementado a calcular el coeficiente de precisión por clase
# Además representa los coeficientes por medio de un histograma y heatmap
################################################################################

#Cargamos los parametros del modelo
LoadModel(circle00)

#Calculamos la precisión del modelo
y_FINAL = Array{Float64}(undef,m)
for i in 1:m
    Zipi = modelo(TI[:,:,:,i:i])
    y_FINAL[i,:] = onecold(Zipi, mClass)
end

Pred = sum(y_FINAL .== Y_rT)/m
print("Precisión: ", round(Pred*100, digits = 2), " %")
