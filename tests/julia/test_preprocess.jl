using Test
using CSV
using DataFrames
include("../../src/julia/preprocess.jl")

@testset "Preprocess Tests" begin
    # Create dummy input file
    df = DataFrame(feature1 = [1.0, 2.0, 3.0])
    CSV.write("../data/raw/test_input.csv", df)
    
    preprocess_data("../data/raw/test_input.csv", "../data/processed/test_output.csv")
    
    # Check output file exists and is normalized
    result = CSV.read("../data/processed/test_output.csv", DataFrame)
    @test size(result, 1) == 3
    @test isapprox(mean(result.feature1), 0.0, atol=1e-5)
end
