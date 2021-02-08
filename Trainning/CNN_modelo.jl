using Flux

function Loss(X,Y)
    Y_hat = modelo(X)
    return crossentropy(Y_hat,Y)
end

#Esta función me permite calcular la entroía cruzada
#Es una versión no mutable de Loss(X,Y)
function mCrossEntropy_1_0(X, Y)
    h,g = size(Y)
    Y_HAT = Array{Float32, 2}(undef,h,g)
    for i in 1:g
        Y_HAT[:,i] = modelo(X[:,:,:,i:i])
    end
    return crossentropy(Y_HAT,Y)
end

# Precisión: 67.42%
# Tiempo: 157.9 minutos
# MODELO PRUEBA 00
modelo = Chain(
    Conv((3,3),1=>64,pad=1,relu),
    MeanPool((2,2)),
    Conv((3,3),64=>128,pad=1,relu),
    MeanPool((2,2)),
    Conv((3,3),128=>256,pad=1,relu),
    MeanPool((2,2)),
    Conv((3,3),256=>512,pad=1,relu),
    MeanPool((2,2)),
    x->reshape(x,:,size(x,4)),
    Dense(4608,11),
    softmax
)
