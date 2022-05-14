using Flux
using Flux.Losses
using Images
import Base.display

include("./fonts/encode.jl")
include("./fonts/stats.jl")


imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));


function loadFolderImages(folderName::String, i)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            #i[1]+=1;
            #println("Folder: ",folderName," imagen ",fileName," patrón número ",i);
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (imageToColorArray.(images), imageToGrayArray.(images), images);
end;


function loadTrainingDataset(aprox2::Bool = false)
    i=[0]
    (positivesColor, positivesGray, positivesRGB) = loadFolderImages("bbdd/positivos",i);
    (negativesColor, negativesGray, negativesRGB) = loadFolderImages("bbdd/negativos",i);
    if aprox2
        (negativesColor2, negativesGray2,  negativesRGB2) = loadFolderImages("bbdd/negativos/aprox2/", i);
        negativesColor = cat(negativesColor,negativesColor2, dims=1)
        negativesGray = cat(negativesGray,negativesGray2, dims=1)
        negativesRGB = cat(negativesRGB,negativesRGB2, dims=1)

    end

    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], [positivesGray; negativesGray], cat(positivesRGB, negativesRGB, dims=1), targets);
end

function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 3, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(28,28,3)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,:,i] .= imagenes[i][:,:,:];
    end;
    return nuevoArray;
end

function redesConvolucionales(images, size = (28,28))
    resize_img(img) = imresize(img, size)
    images = resize_img.(images)
    images = imageToColorArray.(images)
    images = convertirArrayImagenesHWCN(images)
end

function crearRNAConvolucional(topology :: AbstractArray{Tuple{Int64, Int64}}, fPooling = MaxPool, fTransferencia = relu)
    if (length(topology) > 1)
        ann = Chain()

        for (iConv, oConv) in topology[1:(length(topology)-1)]
            ann = Chain(ann..., Conv((3, 3), iConv=>oConv, pad=(1,1), fTransferencia));
            ann = Chain(ann..., fPooling((2,2)));
        end

        ann = Chain(ann..., x -> reshape(x, :, size(x, 4)));

        (iDense, oDense) = topology[length(topology)];
        if (oDense >= 2)
            ann = Chain(ann..., Dense(iDense, oDense), softmax)
        else
            ann = Chain(ann..., Dense(iDense, oDense, σ))
        end

        return ann;
    else
        throw(ErrorException("topology no tiene el tamaño suficiente"));
    end
end

# Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

function entrenarRNAConvolucional(ann, dataset::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,2}},
     test::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,2}}, learningRate :: Real = 0.01, minPrecision :: Real = 0.999)

    opt = ADAM(learningRate);
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);


    println("Comenzando entrenamiento...")
    mejorPrecision = -Inf;
    criterioFin = false;
    numCiclo = 0;
    numCicloUltimaMejora = 0;
    mejorModelo = nothing;
    results = Array{Float32,1}(undef, 0)

    while (!criterioFin)

        # Se entrena un ciclo
        Flux.train!(loss, params(ann), [(dataset[1], dataset[2]')], opt);

        numCiclo += 1;

        # Se calcula la precision en el conjunto de entrenamiento:
        precisionEntrenamiento = accuracy(dataset[2], ann(dataset[1])', 0.5);
        push!(results, precisionEntrenamiento);
        println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            precisionTest = accuracy(test[2], ann(test[1])', 0.5);
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0;
            println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
            numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= minPrecision)
            println("   Se para el entrenamiento por haber llegado a una precision de ", 100*minPrecision ," %")
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end

    return (mejorModelo, results)
end

function entrenarRNAConvolucional(ann, dataset::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,1}},
     test::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,1}}, learningRate :: Real = 0.01, minPrecision :: Real = 0.999)

    trainTargets = reshape(dataset[2], :, 1);
    trainTargets = convert(Array{Bool,2}, trainTargets);

    testTargets = reshape(test[2], :, 1);
    testTargets = convert(Array{Bool,2}, testTargets);

    return entrenarRNAConvolucional(ann, (dataset[1],trainTargets), (test[1],testTargets), learningRate, minPrecision)
end



function main()
    (images, _, imagesRGB, targets) = loadTrainingDataset(true);
    outImages = redesConvolucionales(imagesRGB);

    (nTrain, nTest) = holdOut(size(outImages, 4), 0.2);
    inputsTrain = outImages[:,:,:, nTrain];
    inputsTest = outImages[:,:,:, nTest];
    targetsTrain = targets[nTrain];
    targetsTest = targets[nTest];

    # INPUTS SEN NORMALIZAR

    topologies = [ [(3,16), (16,32), (32,32), (288, 1)],
        [(3,4), (4,8), (8,8), (72, 1)],
        [(3,4), (4,8), (8,16), (144, 1)],
        [(3,8), (8,16), (16,32), (288, 1)],
        [(3,8), (8,16), (16,16), (144, 1)] ]

    for f in [MaxPool, MeanPool]
        for t in topologies
            ann = crearRNAConvolucional(t, MaxPool);
            (ann, results) = entrenarRNAConvolucional(ann, (inputsTrain, targetsTrain), (inputsTest, targetsTest), 0.01, 0.98);

            println("$(f) topology $(t)");
            println("MEAN: $(mean(results)) STD: $(std(results))");

            out = ann(outImages);
            targets = reshape(targets, :, 1);
            cm = confusionMatrix(out', targets, "weighted");
            printStats(cm);
        end
    end
end

main();
