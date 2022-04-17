using DelimitedFiles
using Statistics
using Plots
using Random

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

    (images, _, targets) = loadTrainingDataset(true)
    inputs = extractFeatures(images);
    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    params0 = Dict("transferF" => [], "learningRate" => 0.01, "tValidacion" => 0.2, "maxEpochs" => 1000, "minLoss" => 0, "maxEpochsVal" => 20, "numEntrenamientos" => 10);
    params1 = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1);  #SVM
    params2 = Dict("max_depth" => 4);    #DT
    params3 = Dict("k" => 3);     #kNN

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

function main2()
    (images, _, targets) = loadTrainingDataset()
    inputs = extractFeatures(images);
    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);
    targets = oneHotEncoding(targets);

    topology = [10, 5];
    normalMethod = 0;

    tupla = holdOut(size(inputs, 1), 0.3, 0.2);

    inputsTraining = inputs[tupla[1],:];
    targetsTraining = targets[tupla[1],:];
    if (size(tupla, 1) == 3)
    	inputsValidation = inputs[tupla[2],:];
    	targetsValidation = targets[tupla[2],:];
    	inputsTest = inputs[tupla[3],:];
    	targetsTest = targets[tupla[3],:];
    else
        inputsTest = inputs[tupla[2],:];
    	targetsTest = targets[tupla[2],:];
    end

    if (normalMethod == 1)
        trainParam = calculateZeroMeanNormalizationParameters(inputsTraining);
        normalizeZeroMean!(inputsTraining, trainParam);
        if (size(tupla, 1) == 3)
            normalizeZeroMean!(inputsValidation, trainParam);
        end
        normalizeZeroMean!(inputsTest, trainParam);
    else
        trainParam = calculateMinMaxNormalizationParameters(inputsTraining);
        normalizeMinMax!(inputsTraining, trainParam);
        if (size(tupla, 1) == 3)
      	    normalizeMinMax!(inputsValidation, trainParam);
    	end
        normalizeMinMax!(inputsTest, trainParam);
    end

    tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest, targetsTest), (inputsValidation, targetsValidation), minLoss = 0.1, maxEpochsVal = 100);

    outTest = tupla2[1](inputsTest')';
    cm = confusionMatrix(outTest, targetsTest, "weighted");

    println("\nTopology : $(topology)");
    println("weighted");
    if (normalMethod == 1) println("ZeroMeanNormalization") else println("MinMaxNormalization") end;

    printStats(cm);

    g = plot(1:length(tupla2[2]), tupla2[2], label = "Training");
    plot!(g, 1:length(tupla2[3]), tupla2[3], label = "Validation");
    plot!(g, 1:length(tupla2[4]), tupla2[4], label = "Test");
end

main()
