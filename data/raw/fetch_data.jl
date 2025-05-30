using HTTP
using GZip

function verify_file(file_path::String, expected_size::Int)
    if isfile(file_path)
        size = filesize(file_path)
        return abs(size - expected_size) < 1000  # Allow small variance
    end
    return false
end

function fetch_mnist_data()
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        ("train-images-idx3-ubyte.gz", 9912422),
        ("train-labels-idx1-ubyte.gz", 28881),
        ("t10k-images-idx3-ubyte.gz", 1648877),
        ("t10k-labels-idx1-ubyte.gz", 4542)
    ]
    mkpath("data/raw")
    for (file, expected_size) in files
        output_path = joinpath("data/raw", file)
        if verify_file(output_path, expected_size)
            println("File $file already exists and is valid")
            continue
        end
        try
            url = base_url * file
            HTTP.download(url, output_path)
            if verify_file(output_path, expected_size)
                println("Downloaded and verified: $file")
            else
                error("File $file corrupted")
            end
        catch e
            println("Error downloading $file: $e")
        end
    end
end

fetch_mnist
