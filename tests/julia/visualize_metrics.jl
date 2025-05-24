using Plots
using JSON

function visualize_metrics(metrics_path)
    # Read metrics
    metrics = JSON.parsefile(metrics_path)
    
    # Placeholder: Plot MSE loss
    plot([metrics["mse_loss"]], label="MSE Loss", title="Evaluation Metrics", xlabel="Run", ylabel="Loss")
    savefig("../data/output/metrics/loss_plot.png")
    println("Metrics plot saved to ../data/output/metrics/loss_plot.png")
end

# Example usage 
#please don't use placeholder, we have playground to test 
#the codes. This is for pushing to production 
visualize_metrics("../data/output/metrics/evaluation.json")


