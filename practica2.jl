using DelimitedFiles
using Statistics
using Flux
using Flux.Losses
using Random
using Plots

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})::BitArray
    num_classes = length(classes);
    if (num_classes>2)
        outputs = BitArray{2}(undef,length(feature), num_classes);
        for i in 1:num_classes
            outputs[:,i] = feature.==classes[i];
        end
        outputs;
    else
        reshape(feature.==classes[i], :, 1);
    end
end
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
oneHotEncoding(feature::AbstractArray{Bool,1})::BitArray = reshape(feature, :, 1);

function calculateMinMaxNormalizationParameters(features::AbstractArray{<:Real,2})
    (minimum(features, dims = 1), maximum(features, dims = 1));
end
function calculateZeroMeanNormalizationParameters(features::AbstractArray{<:Real,2})
    (mean(features, dims = 1), std(features, dims = 1));
end

function normalizeMinMax!(inputs::AbstractArray{<:Real,2}, param::NTuple{2, AbstractArray{<:Real,2}})::AbstractArray{<:Real,2}
    for i in 1:length(param[1])
        if param[1][i] == param[2][i]
            inputs[:,i] = zeros(length(inputs[:,i]));
        else
            inputs[:,i] = (inputs[:,i].-param[1][i])/(param[2][i]-param[1][i]);
        end
    end
    inputs;
end
function normalizeMinMax!(inputs::AbstractArray{<:Real,2})::AbstractArray{<:Real,2}
    param = calculateMinMaxNormalizationParameters(inputs);
    normalizeMinMax!(inputs,param);
end
function normalizeMinMax(inputs::AbstractArray{<:Real,2}, param::NTuple{2, AbstractArray{<:Real,2}})::AbstractArray{<:Real,2}
    cp = copy(inputs);
    normalizeMinMax!(cp,param);
end
function normalizeMinMax(inputs::AbstractArray{<:Real,2})::AbstractArray{<:Real,2}
    cp = copy(inputs);
    normalizeMinMax!(cp);
end

function normalizeZeroMean!(inputs::AbstractArray{<:Real,2}, param::NTuple{2, AbstractArray{<:Real,2}})::AbstractArray{<:Real,2}
    for i in 1:length(param[1])
        if param[2][i] == 0
            inputs[:,i] = zeros(length(inputs[:,i]));
        else
            inputs[:,i] = (inputs[:,i].-param[1][i])./param[2][i];
        end
    end
    inputs;
end
function normalizeZeroMean!(inputs::AbstractArray{<:Real,2})::AbstractArray{<:Real,2}
    param = calculateZeroMeanNormalizationParameters(inputs);
    normalizeZeroMean!(inputs,param);
end
function normalizeZeroMean(inputs::AbstractArray{<:Real,2}, param::NTuple{2, AbstractArray{<:Real,2}})::AbstractArray{<:Real,2}
    cp = copy(inputs);
    normalizeZeroMean!(cp,param);
end
function normalizeZeroMean(inputs::AbstractArray{<:Real,2})::AbstractArray{<:Real,2}
    cp = copy(inputs);
    normalizeZeroMean!(cp);
end

function classifyOutputs(outputs::AbstractArray{<:Real,2},threshold=0.5)::BitMatrix
    if(size(outputs,2) == 1)
        outputs.>=threshold;
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        outputs
    end
end

function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1})
    mean(targets.==outputs)
end
function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    if(size(outputs,2) == 1 && size(targets,2) == 1)
        accuracy(targets[:,1], outputs[:,1])
    elseif (size(outputs,2) == size(targets,2))
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims=2);
        mean(correctClassifications);
    else
        throw(DimensionMismatch())
    end
end
function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{<:Real,1}, threshold=0.5)
    accuracy(targets, outputs.>=threshold)
end
function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2}, threshold=0.5)
    if(size(outputs,2) == 1 && size(targets,2) == 1)
        accuracy(targets[:,1],outputs[:,1])
    elseif(size(outputs,2) > 2 && size(targets,2) > 2)
        accuracy(targets, classifyOutputs(outputs, threshold))
    end
end

function crearRNA(topology::AbstractArray{<:Int,1}, entradas::Int64, salidas::Int64, funciones = [])
    ann = Chain();
    numInputsLayer = entradas;
    a = 1;
    for numOutputsLayer in topology
        if funciones == []
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, sigmoid));
        else
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, funciones[a]));
            a = a+1;
        end
        numInputsLayer = numOutputsLayer;

    end
    out_fun = x -> σ.(x);
    if (salidas > 2)
        out_fun = softmax;
    end
    ann = Chain(ann..., Dense(numInputsLayer, salidas, identity), out_fun);
end

function entrenarRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
    ann = crearRNA(topology, size(dataset[1], 2), size(dataset[2], 2));
    vloss = Array{Float32}(undef, 0);

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    for i in 1:maxEpochs
        Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));

        l = loss(dataset[1]', dataset[2]');

        push!(vloss, l);

        if l <= minLoss break; end
    end

    return (ann, vloss);
end

function entrenarRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
    res = reshape(dataset[2], :, 1);
    res = convert(Array{Bool,2}, res);

    return entrenarRNA(topology, (dataset[1],res), maxEpochs, minLoss, learningRate);
end

function entrenarRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2} , AbstractArray{Bool,2}},
     test::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (Array{Real,2}(undef, 0, 0), Array{Bool,2}(undef, 0, 0)), validation::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}} = (Array{Real,2}(undef, 0, 0), Array{Bool,2}(undef, 0, 0)); maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    ann = crearRNA(topology, size(dataset[1], 2), size(dataset[2], 2));
    vector_entrenamiento = Array{Float32, 1}(undef, 0);#=vector donde se guardan los errores de entrenamiento en cada ciclo=#
    vector_validacion = Array{Float32, 1}(undef, 0);#=vector donde se guardan los errores de validacion en cada ciclo=#
    vector_test = Array{Float32, 1}(undef, 0);#=vector donde se guardan los errores de test en cada ciclo=#

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    if (isempty(validation[1]))
        #errores ciclo 0
        trainl = loss(dataset[1]',dataset[2]');
        testl = loss(test[1]',test[2]');
        push!(vector_entrenamiento,trainl);
        push!(vector_test,testl);

    	for i in 1:maxEpochs
        	Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));

        	trainl = loss(dataset[1]',dataset[2]');
            testl = loss(test[1]',test[2]');

            push!(vector_entrenamiento,trainl);
            push!(vector_test,testl);

        	if trainl <= minLoss break; end
    	end
    	return (ann, vector_entrenamiento, vector_validacion, vector_test);
    else
    	epochs = 0;
    	ann_copy = ann;
    	bestValLoss = Inf ;

        #errores ciclo 0
        trainl = loss(dataset[1]',dataset[2]');
        validl = loss(validation[1]',validation[2]');
        testl = loss(test[1]',test[2]');
        push!(vector_entrenamiento,trainl);
        push!(vector_validacion,validl);
        push!(vector_test,testl);
    	for i in 1:maxEpochs
    		if (epochs == maxEpochsVal) break; end #=salimos del bucle si se excedio el maximo numero de ciclos sin mejora=#
    		Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));
            #=calculamos los errores de entrenamiento, validacion y test=#
		    trainl = loss(dataset[1]',dataset[2]');
        	validl = loss(validation[1]',validation[2]');
        	testl = loss(test[1]',test[2]');

        	if(validl < bestValLoss) #=si se mejora el minimo error se guarda la red neuronal que lo ha conseguido=#
        		epochs = 0;
        		bestValLoss = validl;
        		ann_copy = deepcopy(ann);
        	else
        		epochs+=1; #=sino, aumentamos el numero de ciclos sin mejora=#
        	end

            push!(vector_entrenamiento,trainl);
        	push!(vector_validacion,validl);
            push!(vector_test,testl);
            if trainl <= minLoss break; end
    	end
        println(vector_entrenamiento, vector_validacion, vector_test);
    	return (ann_copy, vector_entrenamiento, vector_validacion, vector_test);
    end

end
function entrenarRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2} , AbstractArray{Bool,1}},
    test::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} = (Array{Real,2}(undef, 0, 0), Array{Bool,1}(undef, 0, 0)),
    validation::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}} = (Array{Real,2}(undef, 0, 0), Array{Bool,1}(undef, 0, 0));
    maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    trainTargets = reshape(dataset[2], :, 1);
    trainTargets = convert(Array{Bool,2}, trainTargets);

    testTargets = reshape(test[2], :, 1);
    testTargets = convert(Array{Bool,2}, testTargets);

    validationTargets = reshape(test[2], :, 1);
    validationTargets = convert(Array{Bool,2}, validationTargets);

    return entrenarRNA(topology, (dataset[1],trainTargets), (test[1],testTargets), (validation[1],validationTargets), maxEpochs, minLoss, learningRate, maxEpochsVal)
end


function holdOut(N::Int64, P::Float64)
	train_ind = randperm(N);

	if P>=1.0
		([],train_ind)
	elseif P>0.5
		array= collect(Iterators.partition(train_ind,Int64.(round(N*P, digits=0))));
		(last(array),first(array));
	elseif P!=0.0
		array= collect(Iterators.partition(train_ind,Int64.(round(N*(1-P), digits=0))));
		(first(array),last(array));
	else
		(train_ind, []);
	end
end

function holdOut(N::Int64, Pval::Float64, Ptest::Float64)
	if (Pval+Ptest)<=1.0
		hold1=holdOut(N,Ptest);
		hold2=holdOut(Int64.(length(getfield(hold1,1))), Pval*N/length(getfield(hold1,1)))
		((getfield(hold1,1))[sortperm(getfield(hold2,1))],getfield(hold1,2),(reverse(getfield(hold1,1)))[sortperm(getfield(hold2,2))])
	end
end


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    if size(outputs) == size(targets)
        suma = outputs.+targets;
        positivos = count(i->i==1, targets);
        negativos = size(targets,1) - positivos;
        vn = count(i->i==0, suma); #verdaderos negativos
        vp = count(i->i==2, suma); #verdaderos positivos
        fn = negativos - vn; #falsos negativos
        fp = positivos - vp; #falsos positivos

        sensibilidad = vp / (fn + vp);
        especificidad = vn / (fp + vn);
        vpp = vp / (vp + fp);
        vpn = vn / (vn + fn);
        f1 = 2 / (1/sensibilidad + 1/vpp);

        if vn==size(targets)
            sensibilidad = 1;vpp = 1;
        elseif vp==size(targets)
            especificidad = 1; vpn = 1;
        elseif fn+vp==0 #salida desada positiva
            sensibilidad = 0; vpp = 0; f1 = 0;
        elseif vn+fp==0 #salida desada negativa
            especificidad = 0; vpn = 0;
        end
        (
            (vn+vp) / (vn+vp+fn+fp), #precisión
            (fn+fp) / (vn+vp+fn+fp), #tasa de error
            sensibilidad,
            especificidad,
            vpp, #valor predictivo positivo
            vpn, #valor predictivo negativo
            f1, #F1-score
            [[vn fp]; [fn vp]] #confusion matrix
        )

    else
        throw(DimensionMismatch("Los vectores no tienen la misma longitud"))
    end
end
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, umbral=0.5)
    confusionMatrix(outputs.>=umbral, targets);
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}, opcion::String, umbral=0.5)
    if opcion == "macro" opt = false
    elseif opcion == "weighted" opt = true
    else throw(Exception("Opción no válida.\nOpciones válidas: 'macro' y 'weighted'"));
    end

    cOut = size(outputs, 2); # n classes
    cTar = size(targets, 2);
    iOut = size(outputs, 1); # n instances
    iTar = size(targets, 1);

    if (cOut == cTar && iOut==iTar)
        if cTar == 2
            throw(Exception("Número de columnas incorrecto"));
        elseif cOut == 1
            confusionMatrix(outputs, targets);
        else
            sensibilidad = zeros(cOut);
            especificidad = zeros(cOut);
            vpp = zeros(cOut);
            vpn = zeros(cOut);
            f1 = zeros(cOut);

            for i in 1:cOut
                for n in eachindex(outputs[:,i])
                    if n != 0
                        result = confusionMatrix(outputs[:,i], targets[:,i]);
                        sensibilidad[i] = result[3];
                        especificidad[i] = result[4];
                        vpp[i] = result[5];
                        vpn[i] = result[6];
                        f1[i] = result[7];
                        break;
                    end
                end
            end

            c_matrix = Array{Float64,2}(undef, cOut, cTar);
            acertados = 0;

            for i in 1:cOut
                for j in 1:cTar
                    count = 0;
                    for k in 1:iOut
                        if outputs[k,i] == 1 && targets[k,j] == 1
                            count += 1;
                        end
                    end

                    if i == j acertados += count end
                    c_matrix[i,j] = count;
                end
            end

            acc = accuracy(targets, outputs);

            if opt  # weighted
                return (
                    acc,    # precision
                    1-acc,  # error
                    (sum(sensibilidad)/cOut) * (acertados/iOut),    # sensibilidad
                    (sum(especificidad)/cOut) * (acertados/iOut),   # especificidad
                    (sum(vpp)/cOut) * (acertados/iOut), # vpp
                    (sum(vpn)/cOut) * (acertados/iOut), # vpn
                    (sum(f1)/cOut) * (acertados/iOut),  # f1
                    c_matrix
                );
            else    # macro
                return (
                    acc,    # precision
                    1-acc,  # error
                    sum(sensibilidad)/cOut,     # sensibilidad
                    sum(especificidad)/cOut,    # especificidad
                    sum(vpp)/cOut,  # vpp
                    sum(vpn)/cOut,  # vpn
                    sum(f1)/cOut,   # f1
                    c_matrix
                );
            end
        end
    else
        throw(DimensionMismatch("Las matrices no tienen el mismo tamaño"));
    end
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, opcion::String, umbral=0.5)
    confusionMatrix(classifyOutputs(outputs), targets, opcion, umbral);
end

function confusionMatrix(outputs::AbstractArray{<:Any}, targets::AbstractArray{<:Any}, opcion::String, umbral=0.5)
    @assert (all([in(output, unique(targets)) for output in outputs])) "outputs y targets no tienen las mismas clases";

    confusionMatrix(oneHotEncoding(unique(outputs)), oneHotEncoding(unique(targets)), opcion, umbral);
end


function unoVsTodos(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    numInstances = size(targets, 1);
    numClasses = size(targets, 2);

    outputs = Array{Float32,2}(undef, numInstances, numClasses);

    for numClass in 1:numClasses
        model = entrenarRNA([10], (inputs, targets[:,numClass]))[1];
        outputs[:,numClass] = model(inputs');
    end

    outputs = softmax(outputs')';
    vmax = maximum(outputs, dims=2);
    outputs = (outputs .== vmax);

    return outputs;
end

function crossvalidation(N::Int64, k::Int64)
	vector= collect(1:k);
	vector=repeat(vector,convert(Int64, ceil(N/k)))
	return shuffle!(vector[1:N]);
end

function crossvalidation(targets:: AbstractArray{Bool,2}, k::Int64)
	N=size(targets,1);
	vector= collect(1:N);
	N2=size(targets,2)
	vector2=collect(1:N)

	#for x in N2
	#	if(sum(targets[:,x])<k)
	#		println("pocos")
	#	end
	#end

	for x in 1:N2
		vector2=crossvalidation(sum(targets[:,x]),k)
		i=1
		for y in ((x-1)*N+1):((x-1)*N+N)
			if targets[y]==1
				vector[y-(x-1)*N]=vector2[i]
				i=i+1
			end
		end
	end
	return vector
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
	crossvalidation(oneHotEncoding(targets),k)
end

function modelCrossValidation(model :: int, paremeters :: Dict, inputs :: Array{Any, 2}, targets :: Array{Any, 1}, k :: Int64) 
	resultadoCadaGrupo = collect(1:k);
	index=crossvalidation(targets,k);
	if(model != 0)	
		for x in 1:k
			if(model == 1)   #SVN
				model = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
			elseif(model == 2) #Tree
				model = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);
			elseif(model == 3) #kNN
				model = KNeighborsClassifier(parameters["k"]);
			else
				println("model debe tener un valor de 0 a 3");
			end

			fit!(model, inputs[index.!=x], targets[index.!=x]);
			outgrupoK=predict(model, inputs[index.==x]);   #salidas

			nFilas = length(out);
			resultadoGrupoK= collect(1:nFilas);  #vector cuyos elementos indican si el patrón y coincide en salida y en target
			targetsGrupoK= targets[index.==x];

			for y in 1:nFilas
				resultadoGrupoK[y] = outGrupoK[y]==targetsGrupoK[y];
			end

			aciertos = resultadoGrupoK[resultadoGrupoK.==1];
			resultadoCadaGrupo[x] = length(aciertos)/length(resultadoGrupoK);
		end
		return resultadoCadaGrupo;																							
	else
		targetsOHE = oneHotEncoding(targets);
		results = Array{Float32, 2}(undef, 0);
		for y in 1:k
		    inputsDeIter = inputs[index.!=y];
		targetsDeIter = targets[index.!=y];
				
			tupla = holdOut(size(inputsDeIter, 1), 0.3);
																									
		    inputsTraining = inputsDeIter[tupla[1],:];
    		targetsTraining = targetsDeIter[tupla[1],:];
			inputsValidation = inputsDeIter[tupla[2],:];
    		targetsValidation = targetsDeIter[tupla[2],:];
																									
			tuplaRNA= entrenarRNA(parameters["topology"], (inputsTraining, targetsTraining), (inputs[index.==y], targets[.==y]), (inputsValidation, targetsValidation), 
												  maxEpochs= parameters["maxEpochs"], minLoss= parameters["minLoss"], learningRate= parameters["learningRate"], 
												  maxEpochsVal= parameters["maxEpochsVal"]);
			push!(results, tuplaRNA[3]);
		end
		return results
	end
	
end

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
#    plot!(g, 1:20, tupla2[3], label = "Validation");using DelimitedFiles

#    plot!(g, 1:20, tupla2[4], label = "Test");

    out = unoVsTodos(inputs, targets);

    cm = confusionMatrix(out, targets, "weighted");

    params0 = Dict("topology" => topology, "transferF" => [], "learningRate" => 0.01, "tValidacion" => 0.2, "maxEpochs" => , "minLoss" => , "maxEpochsVal" => );
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
        inputsTraining = normalizeZeroMean!(inputsTraining, trainParam);
        if (size(tupla, 1) == 3)
            inputsValidation = normalizeZeroMean!(inputsValidation, trainParam);
        end
        inputsTest = normalizeZeroMean!(inputsTest, trainParam);
    else
        trainParam = calculateMinMaxNormalizationParameters(inputsTraining);
        inputsTraining = normalizeMinMax!(inputsTraining, trainParam);
        if (size(tupla, 1) == 3)
      	    inputsValidation = normalizeMinMax!(inputsValidation, trainParam);
    	end
        inputsTest = normalizeMinMax!(inputsTest, trainParam);
    end

    tupla2 = entrenarRNA(topology, (inputsTraining, targetsTraining),(inputsTest, targetsTest),(inputsValidation, targetsValidation));

    g = plot(1:length(tupla2[2]), tupla2[2], label = "Training");
    plot!(g, 1:length(tupla2[3]), tupla2[3], label = "Validation");
    plot!(g, 1:length(tupla2[4]), tupla2[4], label = "Test");

    #out = unoVsTodos(inputs, targets);

   # cm = confusionMatrix(out, targets, "weighted");
end

main2()
