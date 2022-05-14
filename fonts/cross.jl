using ScikitLearn

include("encode.jl")
include("rnaOps.jl")
include("stats.jl")

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


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

    if N2>2
        for x in 1:N2
            setindex!(vector,crossvalidation(sum(targets[:,x]),k), findall(i->i==1,targets));
        end
    else
        setindex!(vector,crossvalidation(sum(targets[:]),k), findall(i->i==1,targets));
        setindex!(vector,crossvalidation(N-sum(targets[:]),k), findall(i->i==0,targets));
    end

	return vector
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
	crossvalidation(oneHotEncoding(targets),k)
end

function modelCrossValidation(model :: Symbol, parameters :: Dict, inputs :: AbstractArray{<:Real, 2}, targets :: AbstractArray{<:Real, 1}, k :: Int64)
    @assert(size(inputs,1)==length(targets));

    resultadoCadaGrupo = collect(Float64, 1:k);
	index = crossvalidation(targets,k);

	if (model != :ANN)
		for x in 1:k
			if (model == :SVM)   #SVN
				m = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
			elseif (model == :DT) #Tree
				m = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);
			elseif (model == :KNN) #kNN
				m = KNeighborsClassifier(parameters["k"]);
			else
                throw(ErrorException("model no válido. Utiliza una de estas opciones: :ANN, :SVM, :DT, :KNN"));
				#println("model debe tener un valor entre 0 y 3");
			end

			fit!(m, inputs[index.!=x, :], targets[index.!=x]);
			outGrupoK = predict(m, inputs[index.==x, :]);   #salidas

			nFilas = length(outGrupoK);
			targetsGrupoK = targets[index.==x];

			resultadoCadaGrupo[x] = accuracy(oneHotEncoding(targetsGrupoK), oneHotEncoding(outGrupoK));
		end

		return resultadoCadaGrupo;
	else
		targetsOHE = oneHotEncoding(targets);
		results = Array{Float32, 1}(undef, 0);
		for y in 1:k
		    inputsDeIter = inputs[index.!=y,:];
			targetsDeIter = targets[index.!=y,:];

			#=tupla = holdOut(size(inputsDeIter, 1), parameters["tValidacion"]);

		    inputsTraining = inputsDeIter[tupla[1],:];
    		targetsTraining = targetsDeIter[tupla[1],:];
			inputsValidation = inputsDeIter[tupla[2],:];
    		targetsValidation = targetsDeIter[tupla[2],:];=#
            inputsTest = inputs[index.==y,:];
            targetsTest = targets[index.==y,:];
            metrica = Array{Float32, 1}(undef, 0);

            for i in 1:parameters["numEntrenamientos"]
			    tuplaRNA= entrenarRNA(parameters["topology"], (inputsDeIter, targetsDeIter), (inputsTest, targetsTest),
												  maxEpochs= parameters["maxEpochs"], minLoss= parameters["minLoss"], learningRate= parameters["learningRate"],
												  maxEpochsVal= parameters["maxEpochsVal"]);
                tuplaRNA[1]
                outTest = tuplaRNA[1](inputsTest')'; #salidas
                accuracy(targetsTest, outTest);
                ##cm = confusionMatrix(outTest, targetsTest, "weighted");
                push!(metrica,accuracy(targetsTest, outTest))
            end
			push!(results, mean(metrica));
		end

		return results;
	end
end
