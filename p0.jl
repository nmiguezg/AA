using DelimitedFiles

dataset = readdlm("iris.data",',');

inputs = dataset[:,1:4];
targets = dataset[:,5];
inputs = convert(Array{Float32,2},inputs);
targets = convert(Array{Float32,1},targets);
