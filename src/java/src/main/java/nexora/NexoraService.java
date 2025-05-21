package nexora;

import org.springframework.stereotype.Service;

@Service
public class NexoraService {
    public float[] predict(float[] input) {
        // Placeholder: Load model and run inference
        float[] output = new float[1];
        output[0] = input[0] * 2; // Dummy prediction
        return output;
    }
}
