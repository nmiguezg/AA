using DelimitedFiles
using Statistics
using Plots
using Random
using DataFrames
include("encode.jl")
include("rnaOps.jl")
include("stats.jl")

using JLD2
using Images
import Base.display


# Functions that allow the conversion from images to Float64 arrays
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

# Function to read all of the images in a folder and return them as 2 Float64 arrays: one with color components (3D array) and the other with grayscale components (2D array)
function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (imageToColorArray.(images), imageToGrayArray.(images));
end;

# Functions to load the dataset
function loadTrainingDataset()
    (positivesColor, positivesGray) = loadFolderImages("bbdd/positivos");
    (negativesColor, negativesGray) = loadFolderImages("bbdd/negativos");
    (negativesColor2, negativesGray2) = loadFolderImages("bbdd/negativos/aprox2/");
    negativesColor = cat(negativesColor,negativesColor2, dims=1)
    negativesGray = cat(negativesGray,negativesGray2, dims=1)

    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], [positivesGray; negativesGray], targets);
end;
loadTestDataset() = ((colorMatrix,_) = loadFolderImages("test"); return colorMatrix; );
function extractFeatures(inputs)
    features = zeros(length(inputs),3);
    for i in 1:length(inputs)
        imagen = inputs[i];
        features[i,1] = std(imagen[:,:,1])
        features[i,2] = std(imagen[:,:,2])
        features[i,3] = std(imagen[:,:,3])
    end
    features
end


function compararTopologias()
    Random.seed!(123);

    (images, _, targets) = loadTrainingDataset()
    inputs = extractFeatures(images);
    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);

    params0 = Dict("transferF" => [], "learningRate" => 0.01, "maxEpochs" => 1000, "minLoss" => 0, "maxEpochsVal" => 20, "numEntrenamientos" => 10);

    topologys = [[1,8], [2,9], [3,8], [4,10], [5,8], [6,10], [7,10], [8,10], [9,10], [10,9]]
    
    bMean= 0;
    bTopology=[0];
    for topology in topologys
        params0["topology"] = topology;
        results = modelCrossValidation(:ANN, params0, inputs, targets, 10)
        println(topology," MEAN ", mean(results)," STD: ", std(results))
        if (mean(results)>bMean)
            bMean = mean(results)
            bTopology = topology;
        end
    end
    println(bTopology)

    
    Random.seed!(123);
    for i in 1:10 # kNN
        results = modelCrossValidation(:KNN, Dict("k" => i), inputs, targets, 10)
        println(" k = $(i) \t MEAN: $(mean(results)) STD: $(std(results))")
    end
    
    for i in 1:10 # DT
        results = modelCrossValidation(:DT, Dict("max_depth" => i), inputs, targets, 10)
        println(" depth = $(i) \t MEAN: $(mean(results)) STD: $(std(results))")
    end
    
    Random.seed!(123);
    parameters = [
        Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 9, "C" => 2), 
        Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 7, "C" => 8), 
        Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 4, "C" => 9), 
        Dict("kernel" => "linear", "kernelDegree" => 0, "kernelGamma" => "auto", "C" => 3),
        Dict("kernel" => "linear", "kernelDegree" => 0, "kernelGamma" => "auto", "C" => 7),
        Dict("kernel" => "linear", "kernelDegree" => 0, "kernelGamma" => "auto", "C" => 9),
        Dict("kernel" => "poly", "kernelDegree" => 1, "kernelGamma" => 6, "C" => 2),
        Dict("kernel" => "poly", "kernelDegree" => 1, "kernelGamma" => 6, "C" => 4),
    ]
    for parameter in parameters
        results = modelCrossValidation(:SVM, parameter, inputs, targets, 10)
        if (parameter["kernel"] == "poly")
            println(" kernel = poly degree = $(parameter["kernelDegree"]) gamma = $(parameter["kernelGamma"]) C = $(parameter["C"]) \t\t MEAN: $(mean(results)) STD: $(std(results))")
        elseif (parameter["kernel"] == "rbf")
            println(" kernel = rbf gamma = $(parameter["kernelGamma"]) C = $(parameter["C"]) \t\t MEAN: $(mean(results)) STD: $(std(results))")
        else
            println(" kernel = linear C = $(parameter["C"]) \t\t MEAN: $(mean(results)) STD: $(std(results))")
        end
    end
end


function resultadosMejorTopologia()
    Random.seed!(123);
    (images,_, targets) = loadTrainingDataset()
    inputs = extractFeatures(images);

    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);
    
    parameters = Dict("k" => 9);     #kNN
    m = KNeighborsClassifier(parameters["k"]);
    fit!(m, inputs, targets);
    out = predict(m, inputs);   #salidas
    bCm = confusionMatrix(out, targets); 
    println(parameters)
    printStats(bCm);

    Random.seed!(123);
    parameters = Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 9, "C" => 3);  #SVM
    m = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
    fit!(m, inputs, targets);
    out = predict(m, inputs);   #salidas
    bCm = confusionMatrix(out, targets); 
    println(parameters)
    printStats(bCm);

    Random.seed!(123);
    parameters = Dict("max_depth" => 2);    #DT
    m = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);
    fit!(m, inputs, targets);
    out = predict(m, inputs);   #salidas
    bCm = confusionMatrix(out, targets); 
    println(parameters)
    printStats(bCm);


    Random.seed!(123);
    targets = oneHotEncoding(targets);

    topology = [7,10];

    (iTrain,iTest, iVal) = holdOut(size(inputs, 1), 0., 0.);

    inputsTraining = inputs[iTrain,:];
    targetsTraining = targets[iTrain,:];
   	inputsValidation = inputs[iVal,:];
   	targetsValidation = targets[iVal,:];
   	inputsTest = inputs[iTest,:];
   	targetsTest = targets[iTest,:];
    tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest,targetsTest) ,(inputsValidation, targetsValidation));
    out = tupla2[1](inputs')';
    out = classifyOutputs(out);
    bCm = confusionMatrix(out, targets, "weighted"); 
    println("\nTopology : $(topology)");
    printStats(bCm);
end
#compararTopologias()
resultadosMejorTopologia()

