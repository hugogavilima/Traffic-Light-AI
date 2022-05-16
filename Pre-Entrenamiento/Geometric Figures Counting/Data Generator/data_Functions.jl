using Random

#mFind_Bol recibe un vector a y una constante x
#devuelve false si ningun elemento de a es igual a x
#y true en caso contrario
function mFind_Bol(a, x)
    n = length(a)
    res = false

    for i in 1:n
        if a[i] == x
            res = true
        end
    end
    return res

end

#mCentro recibe los límites de un rectángulo
#        *A = límites del eje X
#        *B = límites del eje Y
#Y devuelve un punto alatorio dentro de esos límtes.
function mPoint(A,B)
    p = (A[2] - A[1])*0.25
    q = (B[2] - B[1])*0.25

    h,k  = rand((A[1]+p:A[2]-p),1)[1], rand((B[1]+q:B[2]-q),1)[1]
    return [h,k]
end


#mCentros recibe un numero natural n, y dos enteros (a, b)
#devuelve los n centros en particiones aleatorias

function mCentros(n, a, b)
    nA, nB = 1, 1

    #encontramos el numero de particiones de a,b
    while nA*nB < n || (nA*nB == 0)
        nA = round(Int, 4*rand(1)[1])
        nB = round(Int, 4*rand(1)[1])
    end
    lA = round(Int, a/nA)
    lB = round(Int, b/nB)

    #escogemos las particiones donde estarán los centros
    points = shuffle(1:nA*nB)[1:n]

    #creamos el vector que almacena los centros
    mC_D = Array{Float32}(undef, 2)
    h0_A, h0_B = 1, 1

    for i in 1:nA
        hf_A = i*lA
        h0_B = 1
        for j in 1:nB
            hf_B = j*lB

            nAB = j + nB*(i-1)
            if mFind_Bol(points, nAB)
                mC_D = hcat(mC_D, mPoint([h0_A, hf_A], [h0_B, hf_B]))
            end
            h0_B = hf_B
        end
        h0_A = hf_A
    end

    return mC_D[:, 2:end]'

end


#mCircles recibe dos enteros (a, b), y número natural n
#Devuelve una imagen de tamaño (a,b) con n circulos de radios aleatorios

function mCircles(a,b,n)
    m = 10000
    A = zeros(a,b)
    t = LinRange(0,2*pi, m)

    #Obtenemos el valor del Radio, que tiene como límites
    #       *Superior: 1/2^(m) veces el lado más pequeño
    #       *Inferior: 1/2^(m+1) veces el lado más pequeño
    #donde m es un número entero
    R = a > b ? b/2^(3) : a/2^(3)
    radio = (R/2)*rand(n) + (R/2)*ones(n)


    center = mCentros(n, a, b)


    for i in 1:n
        p = round.(Int, radio[i]*cos.(t) + center[i, 1]*ones(m))
        q = round.(Int, radio[i]*sin.(t) + center[i, 2]*ones(m))
        for j in 1:m
            if (1 <= p[j] <= a) && (1 <= q[j] <= b)
                    A[p[j], q[j]] = 1
            end
        end
    end

    Gray.(A)
end
