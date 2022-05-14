using DelimitedFiles
using Statistics
using Plots
using Random
using JLD2
using Images
import Base.display

include("./fonts/encode.jl")
include("./fonts/rnaOps.jl")
include("./fonts/stats.jl")
include("./fonts/cross.jl")


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
    return (imageToColorArray.(images), imageToGrayArray.(images), images);
end;

# Functions to load the dataset
function loadTrainingDataset()
    (positivesColor, positivesGray, positivesRGB) = loadFolderImages("bbdd/positivos");
    (negativesColor, negativesGray, negativesRGB) = loadFolderImages("bbdd/negativos");
    (negativesColor2, negativesGray2,  negativesRGB2) = loadFolderImages("bbdd/negativos/aprox2/");
    negativesColor = cat(negativesColor,negativesColor2, dims=1)
    negativesGray = cat(negativesGray,negativesGray2, dims=1)
    negativesRGB = cat(negativesRGB,negativesRGB2, dims=1)
    
    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], [positivesGray; negativesGray], cat(positivesRGB, negativesRGB, dims=1), targets);
end;

function RGBToHSV(imagen)
    rgb_img = imagen
    hsv_img = HSV.(rgb_img)
    channels = channelview(float.(hsv_img))
    (channels[1,:,:], channels[2,:,:], channels[3,:,:])
end

function extractFeaturesHSV(inputs)
    features = zeros(length(inputs),9);
    for i in 1:length(inputs)
        (hue, saturation, value) = inputs[i];
        (height,width) = size(hue)
         h = trunc(Int,height/5)
         w = trunc(Int,width/5)
        features[i,1] = std(hue)
        features[i,2] = std(saturation)
        features[i,3] = std(value)
         #media en la porcion central de la imagen
        features[i,4] = mean(hue[h*2:h*3,w*2:w*3])
        features[i,5] = mean(saturation[h*2:h*3,w*2:w*3])
        features[i,6] = mean(value[h*2:h*3,w*2:w*3])
         #diferencia entre la media en la porcion central y la media en las esquinas
        features[i,7] = features[i,4] - mean([mean(hue[h*0+1:h*1,w*0+1:w*1]),mean(hue[h*4:h*5,w*4:w*5]),mean(hue[h*4:h*5,w*0+1:w*1]), mean(hue[h*0+1:h*1,w*4:w*5])])
        features[i,8] = features[i,5] - mean([mean(saturation[h*0+1:h*1,w*0+1:w*1]),mean(saturation[h*4:h*5,w*4:w*5]),mean(saturation[h*4:h*5,w*0+1:w*1]), mean(saturation[h*0+1:h*1,w*4:w*5])])
        features[i,9] = features[i,6] - mean([mean(value[h*0+1:h*1,w*0+1:w*1]),mean(value[h*4:h*5,w*4:w*5]),mean(value[h*4:h*5,w*0+1:w*1]), mean(value[h*0+1:h*1,w*4:w*5])])
        features[i,7] = abs(features[i,7])
        features[i,8] = abs(features[i,8])
        features[i,9] = abs(features[i,9])
    end
    features
end


function compararTopologias()

    (_, _, imagesRGB, targets) = loadTrainingDataset()
    inputs = extractFeaturesHSV(RGBToHSV.(imagesRGB))
    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);

    params0 = Dict("transferF" => [], "learningRate" => 0.01, "maxEpochs" => 1000, "minLoss" => 0, "maxEpochsVal" => 20, "numEntrenamientos" => 10);

    Random.seed!(123);
    topologys = [[1], [1, 1], [2], [3], [4], [5], [6], [7]]
    

    for topology in topologys
        params0["topology"] = topology;
        results = modelCrossValidation(:ANN, params0, inputs, targets, 10)
        println(topology," MEAN ", mean(results)," STD: ", std(results))
    end
    
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
        Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 3, "C" => 4), 
        Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 4, "C" => 9), 
        Dict("kernel" => "linear", "kernelDegree" => 0, "kernelGamma" => "auto", "C" => 3),
        Dict("kernel" => "linear", "kernelDegree" => 0, "kernelGamma" => "auto", "C" => 7),
        Dict("kernel" => "linear", "kernelDegree" => 0, "kernelGamma" => "auto", "C" => 9),
        Dict("kernel" => "poly", "kernelDegree" => 1, "kernelGamma" => 6, "C" => 2),
        Dict("kernel" => "poly", "kernelDegree" => 4, "kernelGamma" => 6, "C" => 1),
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
    (_, _, imagesRGB, targets) = loadTrainingDataset()
    inputs = extractFeaturesHSV(RGBToHSV.(imagesRGB))

    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);
    
    parameters = Dict("k" => 1);     #kNN
    m = KNeighborsClassifier(parameters["k"]);
    fit!(m, inputs, targets);
    out = predict(m, inputs);   #salidas
    bCm = confusionMatrix(out, targets); 
    println(parameters)
    printStats(bCm);

    Random.seed!(123);
    parameters = Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 3, "C" => 4);  #SVM
    m = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
    fit!(m, inputs, targets);
    out = predict(m, inputs);   #salidas
    bCm = confusionMatrix(out, targets); 
    println(parameters)
    printStats(bCm);

    Random.seed!(123);
    parameters = Dict("max_depth" => 6);    #DT
    m = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);
    fit!(m, inputs, targets);
    out = predict(m, inputs);   #salidas
    bCm = confusionMatrix(out, targets); 
    println(parameters)
    printStats(bCm);


    Random.seed!(123);
    targets = oneHotEncoding(targets);

    topology = [2];

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
compararTopologias()
resultadosMejorTopologia()

