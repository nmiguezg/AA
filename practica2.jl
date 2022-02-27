using DelimitedFiles
using Statistics
using Flux
using Flux.Losses
using Random

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
    out_fun = σ;
    if (salidas > 2)
        out_fun = softmax;
    end
    ann = Chain(ann..., Dense(numInputsLayer, salidas, identity), out_fun);
end

function entrenarRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
    ann = crearRNA(topology, size(dataset[1], 2), size(dataset[2], 2));
    vloss = Array{Float32}(undef, 0);

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    for i in 1:maxEpochs
        Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));

        l = loss(dataset[1]',dataset[2]');
        push!(vloss,l);

        if l <= minLoss break; end
    end

    return (ann, vloss);
end

function entrenarRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
    dataset[2] = reshape(dataset[2], :, 1);

    return entrenarRNA(topology, dataset, maxEpochs, minLoss, learningRate);
end

#=falta que el conjunto de validacion y test sean parametros opcionales que por defecto vengan vacios=#
function entrenarRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, validation::Tuple{AbstractArray{<:Real,2},
AbstractArray{Bool,2}}, test::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    ann = crearRNA(topology, size(dataset[1], 2), size(dataset[2], 2));
    vector_entrenamiento = Array{Float32, 1}(undef, 0);#=vector donde se guardan los errores de entrenamiento en cada ciclo=#
    vector_validacion = Array{Float32, 1}(undef, 0);#=vector donde se guardan los errores de validacion en cada ciclo=#
    vector_test = Array{Float32, 1}(undef, 0);#=vector donde se guardan los errores de test en cada ciclo=#

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    
    if (isempty(validation))
    	for i in 1:maxEpochs
        	Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));

        	l = loss(dataset[1]',dataset[2]');
        	push!(vector_entrenamiento,l);

        	if l <= minLoss break; end
    	end
    	return (ann, vector_entrenamiento, vector_validacion, vector_test);
    else
    	epochs = 0;
    	ann_copy = ann;
    	minloss = 0;
    	for i in 1:maxEpochs
    		if (epochs == maxEpochsVal) break; end #=salimos del bucle si se excedio el maximo numero de ciclos sin mejora=#
    		Flux.train!(loss, params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate));
            #=calculamos los errores de entrenamiento, validacion y test=#
		    trainl = loss(dataset[1]',dataset[2]');
        	validl = loss(validation[1]',validation[2]');
        	testl = loss(test[1]',test[2]');
        	
        	if(validl < minloss) #=si se mejora el minimo error se guarda la red neuronal que lo ha conseguido=#
        		epochs = 0;
        		minloss = validl;
        		ann_copy = deepcopy(ann);
        	else 
        		epochs+=1; #=sino, aumentamos el numero de ciclos sin mejora=#
        	end

            push!(vector_entrenamiento,trainl);
        	push!(vector_validacion,validl);
            push!(vector_test,testl);

        	if validl <= minLoss break; end
    	end
        println(vector_entrenamiento, vector_validacion, vector_test);
    	return (ann_copy, vector_entrenamiento, vector_validacion, vector_test);
    end
    
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


dataset = readdlm("iris.data",',');
inputs = dataset[:,1:4];
targets = dataset[:,5];
@assert (size(inputs,1)==size(targets,1))
inputs = convert(Array{Float32,2},inputs);
targets = oneHotEncoding(targets);

rna = crearRNA([10], size(inputs, 2), size(targets, 2));

trained = entrenarRNA([10], (inputs,targets));

#=
loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
learningRate = 0.01;

ann = Chain(Dense(4,10, sigmoid), Dense(10,3, identity), softmax);

outputs = ann(inputs');

Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate)); =#
