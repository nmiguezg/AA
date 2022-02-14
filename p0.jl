using DelimitedFiles

dataset = readdlm("/home/nico/3º/AA/iris.data",',');

inputs = dataset[:,1:4];
targets = dataset[:,5];
@assert (size(inputs,1)==size(targets,1))
inputs = convert(Array{Float32,2},inputs);

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
function oneHotEncoding(feature::AbstractArray{<:Any,1})::BitArray
    classes = unique(feature);
    oneHotEncoding(feature, classes);
end
oneHotEncoding(feature::AbstractArray{Bool,1})::BitArray = reshape(feature, :, 1);

targets = oneHotEncoding(targets);
using Statistics

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
        if param[1][i] == 0
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
param = calculateMinMaxNormalizationParameters(inputs)
#= 
function classifyOutputs(outputs::AbstractArray{<:Real,2},threshold=0.5)::BitMatrix
    if(size(outputs,2) == 1)
        outputs>=.threshold;
    else
        boolean_outputs = BitArray{2}(undef,size(outputs,1), size(outputs,2));



using Flux.Losses
loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
learningRate = 0.01;

ann = Chain(Dense(4,10, sigmoid), Dense(10,3, identity), softmax);

outputs = ann(inputs');

Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate));  =#