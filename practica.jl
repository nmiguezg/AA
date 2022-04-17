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

# Some functions to display an image stored as Float64 matrix
# Overload the existing display function, either for graysacale or color images
import Base.display
#display(image::Array{Float64,2}) = display(Gray.(image));
#display(image::Array{Float64,3}) = (@assert(size(image,3)==3); display(RGB.(image[:,:,1],image[:,:,2],image[:,:,3])); )

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
function loadTrainingDataset(aprox2::Bool = false)
    (positivesColor, positivesGray) = loadFolderImages("bbdd/positivos");
    (negativesColor, negativesGray) = loadFolderImages("bbdd/negativos");
    if aprox2
        (negativesColor2, negativesGray2) = loadFolderImages("bbdd/negativos/aprox2/");
        negativesColor = cat(negativesColor,negativesColor2, dims=1)
        negativesGray = cat(negativesGray,negativesGray2, dims=1)
    end

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


function main()
    Random.seed!(123);

    (images, _, targets) = loadTrainingDataset()
    inputs = extractFeatures(images);
    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);

    params0 = Dict("transferF" => [], "learningRate" => 0.01, "tValidacion" => 0.2, "maxEpochs" => 1000, "minLoss" => 0, "maxEpochsVal" => 20, "numEntrenamientos" => 10);
    params1 = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1);  #SVM
    params2 = Dict("max_depth" => 4);    #DT
    params3 = Dict("k" => 3);     #kNN
    
    medias = zeros(10,10)
    dt = zeros(10,10)

    for i in 1:10
        for j in 1:10
            topology = [i, j];
            params0["topology"] = topology;

            results = modelCrossValidation(:ANN, params0, inputs, targets, 10)
            medias[i,j] = mean(results)
            dt[i,j] = std(results)
            println(topology," MEAN ",medias[i,j]," STD: ",dt[i,j])
        end
    end
    
    for i in 1:10 # kNN
        results = modelCrossValidation(:KNN, Dict("k" => i), inputs, targets, 10)
        println(" k = $(i) \t MEAN: $(mean(results)) STD: $(std(results))")
    end
    
    # for i in 1:10 # DT
    #     results = modelCrossValidation(:DT, Dict("max_depth" => i), inputs, targets, 10)
    #     println(" depth = $(i) \t MEAN: $(mean(results)) STD: $(std(results))")
    # end

    # for k in ("rbf", "linear", "poly") # SVM
    #     for c in 1:10
    #         if (k == "poly") || (k == "rbf")
    #             for g in 1:10
    #                 if (k == "poly")
    #                     for d in 1:10
    #                         results = modelCrossValidation(:SVM, Dict("kernel" => k, "kernelDegree" => d, "kernelGamma" => g, "C" => c), inputs, targets, 10)
    #                         println(" kernel = $(k) degree = $(d) gamma = $(g) C = $(c) \t\t MEAN: $(mean(results)) STD: $(std(results))")
    #                     end
    #                 else
    #                     results = modelCrossValidation(:SVM, Dict("kernel" => k, "kernelDegree" => 0, "kernelGamma" => g, "C" => c), inputs, targets, 10)
    #                     println(" kernel = $(k) gamma = $(g) C = $(c) \t\t MEAN: $(mean(results)) STD: $(std(results))")
    #                 end
    #             end
    #         else
    #             results = modelCrossValidation(:SVM, Dict("kernel" => k, "kernelDegree" => 0, "kernelGamma" => "auto", "C" => c), inputs, targets, 10)
    #             println(" kernel = $(k) C = $(c) \t\t MEAN: $(mean(results)) STD: $(std(results))")
    #         end
    #     end
    # end

    # for i in 1:10 # ANN
    #     for j in 1:10
    #         local topology = [i, j];
    #         params0["topology"] = topology;
    #
    #         local results = modelCrossValidation(:ANN, params0, inputs, targets, 10)
    #
    #         println(topology," MEAN ",round(mean(results), digits=2)," STD: ",round(std(results), digits=2))
    #     end
    # end
end

#=function main2()
    #Random.seed!(123);
    (images, _, targets) = loadTrainingDataset()
    inputs = extractFeatures(images);
    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);
    targets = oneHotEncoding(targets);

    topology = [7, 2];

    (iTrain,iVal, iTest) = holdOut(size(inputs, 1), 0.2,0.);

    inputsTraining = inputs[iTrain,:];
    targetsTraining = targets[iTrain,:];
   	inputsValidation = inputs[iVal,:];
   	targetsValidation = targets[iVal,:];
   	inputsTest = inputs[iTest,:];
   	targetsTest = targets[iTest,:];

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);
    
    tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest,targetsTest) ,(inputsValidation, targetsValidation));

    out = tupla2[1](inputs')';
    bcm = confusionMatrix(out, targets, "weighted");    
    bAcc=bcm[1];
    for i in 1:100
        tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest,targetsTest) ,(inputsValidation, targetsValidation));

        out = tupla2[1](inputs')';
        cm = confusionMatrix(out, targets, "weighted");
        if (cm[1]>bAcc)
            bAcc = cm[1];
            bcm = cm;
        end

    end
    println("\nTopology : $(topology)");
    printStats(bcm);

    g = plot(1:length(tupla2[2]), tupla2[2], label = "Training");
    plot!(g, 1:length(tupla2[3]), tupla2[3], label = "Validation");
end

main()
