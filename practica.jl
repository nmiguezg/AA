using DelimitedFiles
using Statistics
using Plots

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
function loadTrainingDataset()
    (positivesColor, positivesGray) = loadFolderImages("bbdd/positivos");
    (negativesColor, negativesGray) = loadFolderImages("bbdd/negativos");
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
    targets = dataset[:,5];
    @assert (size(inputs,1)==size(targets,1))
    inputs = convert(Array{Float32,2},inputs);
    targetsOHE = oneHotEncoding(targets);

    topology = [15, 9];
    normalMethod = 1;

#    tupla=holdOut(size(inputs, 1), 0.3,0.2);

#    inputsTraining = inputs[tupla[1],:];
#    targetsTraining = targets[tupla[1],:];
#    if (size(tupla, 1) == 3)
#    	inputsValidation = inputs[tupla[2],:];
#    	targetsValidation = targets[tupla[2],:];
#    	inputsTest = inputs[tupla[3],:];
#    	targetsTest = targets[tupla[3],:];
#    else
#       inputsTest = inputs[tupla[2],:];
#    	targetsTest = targets[tupla[2],:];
#    end

#   if (normalMethod == 1)
#        trainParam = calculateZeroMeanNormalizationParameters(inputsTraining);
#        inputsTraining = normalizeZeroMean!(inputsTraining, trainParam);
#        if (size(tupla, 1) == 3)
#            inputsValidation = normalizeZeroMean!(inputsValidation, trainParam);
#        end
#        inputsTest = normalizeZeroMean!(inputsTest, trainParam);
#    else
#        trainParam = calculateMinMaxNormalizationParameters(inputsTraining);
#        inputsTraining = normalizeMinMax!(inputsTraining, trainParam);
#        if (size(tupla, 1) == 3)
#      	    inputsValidation = normalizeMinMax!(inputsValidation, trainParam);
#    	end
#        inputsTest = normalizeMinMax!(inputsTest, trainParam);
#    end

#    tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining),(inputsTest, targetsTest),(inputsValidation, targetsValidation));

#    g = plot(1:20, tupla2[2], label = "Training");
#    plot!(g, 1:20, tupla2[3], label = "Validation");
#    plot!(g, 1:20, tupla2[4], label = "Test");

    out = unoVsTodos(inputs, targets);

    cm = confusionMatrix(out, targets, "weighted");

    # params0 = Dict("topology" => topology, "transferF" => [], "learningRate" => 0.01, "tValidacion" => 0.2, "maxEpochs" => , "minLoss" => , "maxEpochsVal" => );
    params1 = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1);  #SVM
    params2 = Dict("max_depth" => 4);    #tree
    params3 = Dict("k" => 3);     #kNN

    results = modelCrossValidation(1, params1, inputs, targets, 10)

end

function main2()
    (images,_,targets) = loadTrainingDataset()
    inputs = extractFeatures(images);
    @assert (size(inputs,1)==size(targets,1))
    inputs = convert(Array{Float32,2},inputs);
    targets = oneHotEncoding(targets);

    topology = [15, 9];
    normalMethod = 1;

    tupla=holdOut(size(inputs, 1), 0.3, 0.2);

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

    tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining),(inputsTest, targetsTest),(inputsValidation, targetsValidation));

    g = plot(1:length(tupla2[2]), tupla2[2], label = "Training");
    plot!(g, 1:length(tupla2[3]), tupla2[3], label = "Validation");
    plot!(g, 1:length(tupla2[4]), tupla2[4], label = "Test");

    #out = unoVsTodos(inputs, targets);

   # cm = confusionMatrix(out, targets, "weighted");
end

main2()
