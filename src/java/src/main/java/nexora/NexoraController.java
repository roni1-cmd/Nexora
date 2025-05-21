package nexora;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class NexoraController {
    private final NexoraService service;

    public NexoraController(NexoraService service) {
        this.service = service;
    }

    @PostMapping("/predict")
    public float[] predict(@RequestBody float[] input) {
        return service.predict(input);
    }
}
