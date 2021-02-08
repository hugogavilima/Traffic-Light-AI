using BSON: @save, @load
using Flux, Zygote
#Fucion determinamos las clases de un vector
function mClases(P)
    m = length(P)
    mC = [P[1]]
    for i in 1:m
        if sum(mC .== P[i]) == 0
            mC = vcat(mC, P[i])
        end
    end
    return mC
end

#Función matriz de canal de una imagen
#Recibe una imagen IMG, y el color a extraer wcolor
function mChannelImg(IMG)
    p,q = size(IMG)
    T = Array{Float32}(undef, p, q, 1)
    for i in 1:p
        for j in 1:q
            mPix = IMG[i,j]
            T[i,j,1] = mPix.val
        end
    end
    return T
end

#Guarda los parmatros del modelo
function SaveModel(number)
    weights = params(modelo)
    @save ("/home/aquilesbailo/Traffic-Light-AI/Trainning/modelos/modelo_"*number*".bson") weights
end

#Carga los parametros del modelo
function LoadModel(number)
    @load ("/home/aquilesbailo/Traffic-Light-AI/Trainning/modelos/modelo_"*number*".bson") weights
    Flux.loadparams!(modelo, weights)
end
