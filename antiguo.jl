using DelimitedFiles
using Statistics
using Flux
using Flux.Losses


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if size(classes, 1) > 2
        matrix = Array{Bool}(undef, size(feature, 1), size(classes, 1));

        for i in 1:size(classes, 1)
            matrix[:,i] = feature.==classes[i];
        end

        return matrix;
    else
        matrix = Array{Bool}(undef, size(feature, 1), 1);

        matrix = feature.==classes[1];

        return reshape(matrix, :, 1);
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);


calculateMinMaxNormalizationParameters(matrix::AbstractArray{<:Real,2}) = (minimum(matrix, dims=1), maximum(matrix, dims=1));


calculateZeroMeanNormalizationParameters(matrix::AbstractArray{<:Real,2}) = (mean(matrix, dims=1), std(matrix, dims=1));


function normalizeMinMax!(matrix::AbstractArray{<:Real,2}, params::NTuple{2, AbstractArray{<:Real,2}})
    for i in 1:size(matrix, 2)
        matrix[:,i] = (matrix[:,i].-params[1][i]) / (params[2][i] - params[1][i]);
    end
end

normalizeMinMax!(matrix::AbstractArray{<:Real,2}) = normalizeMinMax!(matrix, calculateMinMaxNormalizationParameters(matrix));


function normalizeMinMax(matrix::AbstractArray{<:Real,2}, params::NTuple{2, AbstractArray{<:Real,2}})
    m2 = copy(matrix);

    normalizeMinMax!(m2, params);

    return m2;
end

function normalizeMinMax(matrix::AbstractArray{<:Real,2})
    m2 = copy(matrix);

    normalizeMinMax!(m2);

    return m2;
end


function normalizeZeroMean!(matrix::AbstractArray{<:Real,2}, params::NTuple{2, AbstractArray{<:Real,2}})
    for i in 1:size(matrix, 2)
        matrix[:,i] = (matrix[:,i].-params[1][i]) / params[2][i];
    end
end

normalizeZeroMean!(matrix::AbstractArray{<:Real,2}) = normalizeZeroMean!(matrix, calculateZeroMeanNormalizationParameters(matrix));

function normalizeZeroMean(matrix::AbstractArray{<:Real,2}, params::NTuple{2, AbstractArray{<:Real,2}})
    m2 = copy(matrix);

    normalizeZeroMean!(m2, params);

    return m2;
end

function normalizeZeroMean(matrix::AbstractArray{<:Real,2})
    m2 = copy(matrix);

    normalizeZeroMean!(m2);

    return m2;
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold::Float64=0.5)
    if size(outputs, 1) < 1
        return outputs;
    elseif size(outputs, 1) == 1
        return outputs.>=threshold;
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs;
    end
end


dataset = readdlm("iris.data", ',');
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];

inputs = dataset[:, 1:4];
targets = dataset[:, 5];

inputs = convert(Array{Float32,2}, inputs);
targets = convert(Array{String,1}, targets);

targets = oneHotEncoding(targets, classes);


rna = Chain(Dense(4, 3, σ), Dense(3, 3, identity), softmax);

outputs = rna(inputs');

out = classifyOutputs(outputs');

#
# loss(x, ideal) = Losses.crossentropy(rna(x), ideal);
#
# learningRate = 0.01;
#
# # Flux.train!(loss, params(rna), [(inputs', targets')], ADAM(learningRate));
#
# for i in 1:10
#     Flux.train!(loss, params(rna), [(inputs', targets')], ADAM(learningRate));
# end
