using Statistics
using Random
using ScikitLearn

include("encode.jl")
include("rnaOps.jl")

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1})
    mean(targets.==outputs)
end
function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    if (size(outputs,2) == 1 && size(targets,2) == 1)
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
    if (size(outputs,2) == 1 && size(targets,2) == 1)
        accuracy(targets[:,1],outputs[:,1])
    elseif (size(outputs,2) > 2 && size(targets,2) > 2)
        accuracy(targets, classifyOutputs(outputs, threshold))
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
            confusionMatrix(outputs[:,1], targets[:,1], umbral);
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

function printStats(cm :: Tuple{<:Real, <:Real, <:Real, <:Real, <:Real, <:Real, <:Real, AbstractArray{Int64, 2}})
    println("\n    Precisión : $(cm[1])");
    println("        Error : $(cm[2])");
    println(" Sensibilidad : $(cm[3])");
    println("Especificidad : $(cm[4])");
    println("          VPP : $(cm[5])");
    println("          VPN : $(cm[6])");
    println("           F1 : $(cm[7])\n");

    println("Matriz de confusión :")
    for l in 1:size(cm[8], 1) # Imprime matriz de confusión
        println("\t $(cm[8][l,:])")
    end
end
