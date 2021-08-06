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

# # Tiempo: 120.11 minutos
# # MODELO circle00
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MeanPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MeanPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MeanPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MeanPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4608,11),
#     softmax
# )

# # Tiempo: 132.84 minutos
# # MODELO circle01
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4608,11),
#     softmax
# )

# # Tiempo: 184.41 minutos
# # MODELO circle02
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1),
#     BatchNorm(64, relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1),
#     BatchNorm(128, relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>128,pad=1),
#     BatchNorm(128, relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4608,11),
#     softmax
# )

# # Tiempo: 101.68 minutos
# # MODELO circle03
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(256, 11),
#     BatchNorm(11, relu),
#     softmax
# )

# # Tiempo: 167.44 minutos
# # MODELO circle04
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4608,11),
#     BatchNorm(11, relu),
#     softmax
# )

# # Tiempo: 172.45 minutos
# # MODELO circle05
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),512=>1024,pad=2,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4096,11),
#     BatchNorm(11, relu),
#     softmax
# )

# # Tiempo: 129.46 minutos
# # MODELO circle06
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4608,11),
#     softmax
# )

# # Tiempo: 159.39 minutos
# # MODELO circle07
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),512=>1024,pad=2,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4096,11),
#     softmax
# )


# # Tiempo: 185.24 minutos
# # MODELO circle08
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),512=>1024,pad=2,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4096,11),
#     BatchNorm(11, relu),
#     softmax
# )


# Tiempo: 185.24 minutos
# MODELO circle08_P1
modelo = Chain(
    Conv((3,3),1=>64,pad=1,relu),
    MaxPool((2,2)),
    Conv((3,3),64=>128,pad=1,relu),
    MaxPool((2,2)),
    Conv((3,3),128=>256,pad=1,relu),
    MaxPool((2,2)),
    Conv((3,3),256=>512,pad=1,relu),
    MaxPool((2,2)),
    Conv((3,3),512=>1024,pad=2,relu),
    MaxPool((2,2)),
    x->reshape(x,:,size(x,4)),
    Dense(4096,10),
    BatchNorm(10, relu),
    softmax
)


# # Tiempo: 90.46 minutos
# # MODELO circle09
# modelo = Chain(
#     Conv((3,3),1=>64,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),64=>128,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),128=>256,pad=1,relu),
#     MaxPool((2,2)),
#     Conv((3,3),256=>512,pad=1,relu),
#     MaxPool((2,2)),
#     x->reshape(x,:,size(x,4)),
#     Dense(4608,11),
#     BatchNorm(11, relu),
#     softmax
# )
