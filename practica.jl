using DelimitedFiles
using Statistics
using Plots
using Random
using Images
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

# Functions to load the dataset
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
end;
loadTestDataset() = ((colorMatrix,_,_) = loadFolderImages("test"); return colorMatrix; );
function extractFeatures(inputs)
    features = zeros(length(inputs),9);
    for i in 1:length(inputs)
        imagen = inputs[i];
        (height,width) = size(inputs[i])
         h = trunc(Int,height/5)
         w = trunc(Int,width/5)
        features[i,1] = std(imagen[:,:,1])
        features[i,2] = std(imagen[:,:,2])
        features[i,3] = std(imagen[:,:,3])
        #media en la porcion central de la imagen
        features[i,4] = mean(imagen[h*2:h*3,w*2:w*3,1])
        features[i,5] = mean(imagen[h*2:h*3,w*2:w*3,2])
        features[i,6] = mean(imagen[h*2:h*3,w*2:w*3,3])
        #diferencia entre la media en la porcion central y la media en las esquinas
        features[i,7] = features[i,4] - mean([mean(imagen[h*0+1:h*1,w*0+1:w*1,1]),mean(imagen[h*4:h*5,w*4:w*5,1]),mean(imagen[h*4:h*5,w*0+1:w*1,1]), mean(imagen[h*0+1:h*1,w*4:w*5,1])])
        features[i,8] = features[i,5] - mean([mean(imagen[h*0+1:h*1,w*0+1:w*1,2]),mean(imagen[h*4:h*5,w*4:w*5,2]),mean(imagen[h*4:h*5,w*0+1:w*1,2]), mean(imagen[h*0+1:h*1,w*4:w*5,2])])
        features[i,9] = features[i,6] - mean([mean(imagen[h*0+1:h*1,w*0+1:w*1,3]),mean(imagen[h*4:h*5,w*4:w*5,3]),mean(imagen[h*4:h*5,w*0+1:w*1,3]), mean(imagen[h*0+1:h*1,w*4:w*5,3])])
        features[i,7] = abs(features[i,7])
        features[i,8] = abs(features[i,8])
        features[i,9] = abs(features[i,9])
    end
    features
end
function extractFeaturesHSV(inputs)
    features = zeros(length(inputs),9);
    for i in 1:length(inputs)
        (hue,saturation,value) = inputs[i];
        (height,width) = size(hue)
         h = trunc(Int,height/5)
         w = trunc(Int,width/5)
        features[i,1] = std(hue)
        features[i,2] = std(saturation)
        features[i,3] = std(value)
        #= #media en la porcion central de la imagen
        features[i,4] = mean(imagen[h*2:h*3,w*2:w*3,1])
        features[i,5] = mean(imagen[h*2:h*3,w*2:w*3,2])
        features[i,6] = mean(imagen[h*2:h*3,w*2:w*3,3])
        #diferencia entre la media en la porcion central y la media en las esquinas
        features[i,7] = features[i,4] - mean([mean(imagen[h*0+1:h*1,w*0+1:w*1,1]),mean(imagen[h*4:h*5,w*4:w*5,1]),mean(imagen[h*4:h*5,w*0+1:w*1,1]), mean(imagen[h*0+1:h*1,w*4:w*5,1])])
        features[i,8] = features[i,5] - mean([mean(imagen[h*0+1:h*1,w*0+1:w*1,2]),mean(imagen[h*4:h*5,w*4:w*5,2]),mean(imagen[h*4:h*5,w*0+1:w*1,2]), mean(imagen[h*0+1:h*1,w*4:w*5,2])])
        features[i,9] = features[i,6] - mean([mean(imagen[h*0+1:h*1,w*0+1:w*1,3]),mean(imagen[h*4:h*5,w*4:w*5,3]),mean(imagen[h*4:h*5,w*0+1:w*1,3]), mean(imagen[h*0+1:h*1,w*4:w*5,3])])
        features[i,7] = abs(features[i,7])
        features[i,8] = abs(features[i,8])
        features[i,9] = abs(features[i,9])=#
    end
    features
end



function main()
    Random.seed!(123);

    (images, _, imagesRGB, targets) = loadTrainingDataset(true)
    inputs = extractFeatures(images);
    inputs = extractFeaturesHSV(RGBToHSV(imagesRGB))
    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);

    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);

    params0 = Dict("transferF" => [], "learningRate" => 0.01, "maxEpochs" => 1000, "minLoss" => 0, "maxEpochsVal" => 20, "numEntrenamientos" => 50);
    params1 = Dict("kernel" => "rbf", "kernelDegree" => 3, "kernelGamma" => 2, "C" => 1);  #SVM
    params2 = Dict("max_depth" => 4);    #DT
    params3 = Dict("k" => 3);     #kNN
    
    bMean= 0;
    bTopology=[0];
    for i in 1:10
        for j in 0:10
            topology = [i, j];
            if j==0  
                topology = [i];
            end
            params0["topology"] = topology;

            results = modelCrossValidation(:ANN, params0, inputs, targets, 10)
            println(topology," MEAN ", mean(results)," STD: ", std(results))
            if (mean(results)>bMean)
                bMean = mean(results)
                bTopology = topology;
            end
        end
    end
    println(bTopology)
    #=
    for i in 1:10 # kNN
        results = modelCrossValidation(:KNN, Dict("k" => i), inputs, targets, 10)
        println(" k = $(i) \t MEAN: $(mean(results)) STD: $(std(results))")
    end
    
    for i in 1:10 # DT
         results = modelCrossValidation(:DT, Dict("max_depth" => i), inputs, targets, 10)
         println(" depth = $(i) \t MEAN: $(mean(results)) STD: $(std(results))")
     end
    bMean= 0;
    bSVM = Dict("kernel" => "", "kernelDegree" => 0, "kernelGamma" => 0, "C" => 0);
       for k in ("rbf", "linear", "poly") # SVM
         for c in 1:10
             if (k == "poly") || (k == "rbf")
                 for g in 1:10
                     if (k == "poly")
                         for d in 1:10
                            results = modelCrossValidation(:SVM, Dict("kernel" => k, "kernelDegree" => d, "kernelGamma" => g, "C" => c), inputs, targets, 10)
                            println(" kernel = $(k) degree = $(d) gamma = $(g) C = $(c) \t\t MEAN: $(mean(results)) STD: $(std(results))")
                            if (mean(results)>bMean)
                                bMean = mean(results)
                                bSVM = Dict("kernel" => k, "kernelDegree" => d, "kernelGamma" => g, "C" => c);
                            end
                         end
                     else
                         results = modelCrossValidation(:SVM, Dict("kernel" => k, "kernelDegree" => 0, "kernelGamma" => g, "C" => c), inputs, targets, 10)
                         println(" kernel = $(k) gamma = $(g) C = $(c) \t\t MEAN: $(mean(results)) STD: $(std(results))")
                         if (mean(results)>bMean)
                            bMean = mean(results)
                            bSVM = Dict("kernel" => k, "kernelDegree" => 0, "kernelGamma" => g, "C" => c);
                        end
                    end
                 end
             else
                 results = modelCrossValidation(:SVM, Dict("kernel" => k, "kernelDegree" => 0, "kernelGamma" => "auto", "C" => c), inputs, targets, 10)
                 println(" kernel = $(k) C = $(c) \t\t MEAN: $(mean(results)) STD: $(std(results))")
                 if (mean(results)>bMean)
                    bMean = mean(results)
                    bSVM = Dict("kernel" => k, "kernelDegree" => 0, "kernelGamma" => "auto", "C" => c);
                end
             end
         end
    end
    print(bSVM)=#
end

function main2()
    Random.seed!(123);

    (images, _, imagesRGB, targets) = loadTrainingDataset(true)
    inputs = extractFeatures(images);
    inputs = extractFeaturesHSV(RGBToHSV.(imagesRGB))

    @assert (size(inputs,1) == size(targets,1))
    inputs = convert(Array{Float32,2}, inputs);
    targets = oneHotEncoding(targets);

    topology = [3];

    (iTrain,iTest, iVal) = holdOut(size(inputs, 1), 0., 0.);

    inputsTraining = inputs[iTrain,:];
    targetsTraining = targets[iTrain,:];
   	inputsValidation = inputs[iVal,:];
   	targetsValidation = targets[iVal,:];
   	inputsTest = inputs[iTest,:];
   	targetsTest = targets[iTest,:];


    trainParam = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs, trainParam);

    #parameters = Dict("max_depth" => 6);    #DT
    parameters = Dict("kernel" => "poly", "kernelDegree" => 6, "kernelGamma" => 4, "C" => 1);  #SVM

    #parameters = Dict("k" => 5);     #kNN

    #m = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);
    #m = KNeighborsClassifier(parameters["k"]);

    m = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
    fit!(m, inputs, targets);
    out = predict(m, inputs);   #salidas
    #=acc= 0;
    tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest,targetsTest) ,(inputsValidation, targetsValidation));
    out = tupla2[1](inputs')';
    out = classifyOutputs(out);=#
    bCm = confusionMatrix(out, targets, "weighted"); 
    #=for i in 1:50
        tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining), (inputsTest,targetsTest) ,(inputsValidation, targetsValidation));

        out = tupla2[1](inputs')';
        out = classifyOutputs(out);
        cm = confusionMatrix(out, targets, "weighted");  
        if (cm[1]>acc)
            acc= cm[1];
            bCm = cm;
        end
    end=#
    #println("\nTopology : $(topology)");
    println(parameters)
    printStats(bCm);
    #println(findall(i->i!=1,out.==targets))
end

#main2()
function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 3, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(28,28,3)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,:,i] .= imagenes[i][:,:,:];
    end;
    return nuevoArray;
end;

function redesConvulacionales(images, size = (28,28))
    resize_img(img) = imresize(img, size)
    images = resize_img.(images)
    images = imageToColorArray.(images)
    images = convertirArrayImagenesHWCN(images)
end
function RGBToHSV(imagen)
    rgb_img = imagen
    hsv_img = HSV.(rgb_img)
    channels = channelview(float.(hsv_img))
    (channels[1,:,:], channels[2,:,:], channels[3,:,:])
end
function HSVMask(imagen)
    mask = zeros(size(hue_img))
    h, s, v = 100, 50, 150
    for ind in eachindex(hue_img)
        if saturation_img[ind] <= s/255
        mask[ind] = 1
        end
    end
    matrizBooleana = colorview(Gray, mask)
    labelArray = ImageMorphology.label_components(matrizBooleana);
    tamanos = ImageMorphology.component_lengths(labelArray);
    boundingBoxes = ImageMorphology.component_boxes(labelArray);
    tamanos = ImageMorphology.component_lengths(labelArray)./length(imagen);
    centroides = ImageMorphology.component_centroids(labelArray);
    etiquetasEliminar = cat(findall(tamanos .<= 0.05) .- 1, findall(tamanos .>= 0.95) .- 1, dims=1); # Importate el -1, porque la primera etiqueta es la 0
    matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
    display(Gray.(matrizBooleana));
    labelArray = ImageMorphology.label_components(matrizBooleana);
    centroides = ImageMorphology.component_centroids(labelArray);


    imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);
    # Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
    centroides = ImageMorphology.component_centroids(labelArray)[2:end];
    # Para cada centroide, ponemos su situacion en color rojo
    for centroide in centroides
        x = Int(round(centroide[1]));
        y = Int(round(centroide[2]));
        imagenObjetos[ x, y ] = RGB(1,0,0);
    end;
    
    boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
    for boundingBox in boundingBoxes
        x1 = boundingBox[1][1];
        y1 = boundingBox[1][2];
        x2 = boundingBox[2][1];
        y2 = boundingBox[2][2];
        imagenObjetos[ x1:x2 , y1 ] .= RGB(0,1,0);
        imagenObjetos[ x1:x2 , y2 ] .= RGB(0,1,0);
        imagenObjetos[ x1 , y1:y2 ] .= RGB(0,1,0);
        imagenObjetos[ x2 , y1:y2 ] .= RGB(0,1,0);
    end;
    (imagenObjetos);
end