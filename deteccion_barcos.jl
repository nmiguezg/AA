#!/usr/bin/env julia

using Statistics
using Random
using Images
using JLD2
using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


include("./fonts/encode.jl")
include("./fonts/rnaOps.jl")

# Functions that allow the conversion from images to Float64 arrays
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
    return (imageToColorArray.(images), images);
end;

# Functions to load the dataset
function loadTrainingDataset()
    (positivesColor, positivesRGB) = loadFolderImages("bbdd/positivos");
    (negativesColor, negativesRGB) = loadFolderImages("bbdd/negativos");
    (negativesColor2,  negativesRGB2) = loadFolderImages("bbdd/negativos/aprox2/");
    negativesColor = cat(negativesColor,negativesColor2, dims=1)
    negativesRGB = cat(negativesRGB,negativesRGB2, dims=1)

    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], cat(positivesRGB, negativesRGB, dims=1), targets);
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



function trainModel(model::String)
    (_, imagesRGB, targets) = loadTrainingDataset()
    inputs = extractFeaturesHSV(RGBToHSV.(imagesRGB))

    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);
    if model == "KNN"
        parameters = Dict("k" => 1);     #kNN
        m = KNeighborsClassifier(parameters["k"]);
        fit!(m, inputs, targets);
        ((:KNN, m,), trainParam)
    elseif model == "SVM"
        parameters = Dict("kernel" => "rbf", "kernelDegree" => 0, "kernelGamma" => 3, "C" => 4);  #SVM
        m = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
        fit!(m, inputs, targets);
        ((:SVM, m,), trainParam)
    elseif model == "DT"
        parameters = Dict("max_depth" => 6);    #DT
        m = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);
        fit!(m, inputs, targets);
        ((:DT, m,), trainParam)
    else
        targets = oneHotEncoding(targets);

        topology = [3];
    
        (iTrain,iTest, iVal) = holdOut(size(inputs, 1), 0., 0.);
    
        inputsTraining = inputs[iTrain,:];
        targetsTraining = targets[iTrain,:];
           inputsValidation = inputs[iVal,:];
           targetsValidation = targets[iVal,:];
           inputsTest = inputs[iTest,:];
           targetsTest = targets[iTest,:];
        tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest,targetsTest) ,(inputsValidation, targetsValidation));
        #=
        out = tupla2[1](inputs')';
        out = classifyOutputs(out);
        bCm = confusionMatrix(out, targets, "weighted");
        for i in 1:50
            tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest,targetsTest) ,(inputsValidation, targetsValidation));

            out = tupla2[1](inputs')';
            out = classifyOutputs(out);
            cm = confusionMatrix(out, targets, "weighted");
            if (cm[1]>acc)
                acc= cm[1];
                bCm = cm;
            end
        end =#
        ((:ANN,  tupla2[1],), trainParam)
    end
end



function detectBoat(image, model, trainParam, initialSize = 5, maxSize = 30, step = 5, rHeight= 100, rWidth = 100)
    (height,width) = size(image)
    coordenadasBarco = zeros(Int,rHeight,rWidth)
    imResized = imresize(image,(rHeight,rWidth))
    for wsh in initialSize:step:maxSize
        for wsx in initialSize:step:maxSize
            for h in 1:step:(rHeight-wsh)
                for x in 1:step:(rWidth-wsx)
                    inputs = extractFeaturesHSV([RGBToHSV(imResized[h:wsh+h, x:wsx+x])])
                    normalizeMinMax!(inputs, trainParam);
                    if model[1] == :ANN
                        out = (model[2])(inputs')';
                        out = classifyOutputs(out);
                    else
                        out = predict(model[2],inputs);
                    end
                    for i in out
                        if i==1
                            coordenadasBarco[h:h+wsh, x:x+wsx].+=1
                        end
                    end
                end
            end;
        end
    end
    coordenadasBarco = coordenadasBarco.>(trunc(Int,maximum(coordenadasBarco)/3))
    labelArray = ImageMorphology.label_components(coordenadasBarco);
            boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
            for boundingBox in boundingBoxes
                x1 = trunc(Int, boundingBox[1][1]*(height/rHeight));
                y1 = trunc(Int,boundingBox[1][2]*(width/rWidth));
                x2 = trunc(Int,boundingBox[2][1]*(height/rHeight));
                y2 = trunc(Int,boundingBox[2][2]*(width/rWidth));
                image[ x1:x2 , y1 ] .= RGB(0,1,0);
                image[ x1:x2 , y2 ] .= RGB(0,1,0);
                image[ x1 , y1:y2 ] .= RGB(0,1,0);
                image[ x2 , y1:y2 ] .= RGB(0,1,0);
            end
    image
end

@assert(length(ARGS)>1)

function readArgs()
    model="";
    image="";
    outputImage="output.png";
    for i in 1:length(ARGS)
        if ARGS[i]=="-model"
            model = ARGS[i+1]
        elseif ARGS[i]=="-image"
            image = ARGS[i+1]
        elseif ARGS[i]=="-output"
            outputImage = ARGS[i+1]
        end
    end
    (model, image, outputImage)
end

(model,image, outputImage) = readArgs()
@assert(model in ["RNA","KNN", "SVM", "DT"] )
(m, trainParam) = trainModel(model);


@assert(image!="")
image = load(image);
# Check that they are color images
@assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))

image = detectBoat(image, m,trainParam)
save(outputImage, image) 


