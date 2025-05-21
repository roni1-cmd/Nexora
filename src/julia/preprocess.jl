using CSV
using DataFrames

function preprocess_data(input_path, output_path)
    # Read raw data
    df = CSV.read(input_path, DataFrame)
    
    # Placeholder: Normalize data
    df[!, :feature1] = (df[!, :feature1] .- mean(df[!, :feature1])) ./ std(df[!, :feature1])
    
    # Save processed data
    CSV.write(output_path, df)
    println("Data preprocessed and saved to $output_path")
end

# Example usage
preprocess_data("../data/raw/input.csv", "../data/processed/processed.csv")
