using Flux
using Flux.Losses

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
     test::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (Array{Real,2}(undef, 0, 0), Array{Bool,2}(undef, 0, 0)), validation::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}} = (Array{Real,2}(undef, 0, 0), Array{Bool,2}(undef, 0, 0));
     maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20)

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

        	if (validl < bestValLoss) #=si se mejora el minimo error se guarda la red neuronal que lo ha conseguido=#
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
        #println(vector_entrenamiento, vector_validacion, vector_test);
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
