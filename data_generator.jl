using Images
using DataFrames, CSV

#Función generadora de las imágenes
function mCircles(a,b,n; grosor = 0)
    m = 10000
    A = zeros(a,b)
    t = LinRange(0,2*pi, m)

    center = zeros(n,2)
    center[:, 1] = rand((1:a),n)
    center[:, 2] = rand((1:b),n)

    R = a > b ? b/4 : a/4
    radio = R*rand(n)

    for i in 1:n
        p = round.(Int, radio[i]*cos.(t) + center[i, 1]*ones(m))
        q = round.(Int, radio[i]*sin.(t) + center[i, 2]*ones(m))
        for j in 1:m
            for k in 0:grosor
                if (1 <= p[j] <= a) && (1 <= q[j] <= b)
                    A[p[j], q[j]] = 1
                end
                if (1 <= p[j]+k <= a) && (1 <= q[j] <= b)
                    A[p[j]+k, q[j]] = 1
                end
                if (1 <= p[j] <= a) && (1 <= q[j]+k <= b)
                    A[p[j], q[j]+k] = 1
                end
                if (1 <= p[j]-k <= a) && (1 <= q[j] <= b)
                    A[p[j]-k, q[j]] = 1
                end
                if (1 <= p[j] <= a) && (1 <= q[j]-k <= b)
                    A[p[j], q[j]-k] = 1
                end
            end

        end
    end

    Gray.(A)
end

#Definimos la ruta de los archivos de entrenamiento
str_train = "/home/aquilesbailo/Traffic-Light-AI/mlight/"

IND = zeros(10999)
train = zeros(10999)

for i in 1:10999

    #Obtenemos la clase de la imagen
    p = convert(Int64, floor(i/1000))

    #Creamos la imagen
    img  = mCircles(50,50,p, grosor = 0)
    save(str_train*string(i)*".jpg",img)

    #Salvamos la clase de cad imagen
    IND[i] = i
    train[i] = p
end

#Salvamos los indices de las imagenes
Table = DataFrame(Ind = IND, TRAIN = train)
CSV.write("train_data.csv", Table);