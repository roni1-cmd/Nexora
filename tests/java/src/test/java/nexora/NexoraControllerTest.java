package nexora;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

@SpringBootTest
public class NexoraControllerTest {
    @Autowired
    private NexoraController controller;

    @Test
    void testPredict() {
        float[] input = {1.0f, 2.0f, 3.0f};
        float[] expected = {2.0f, 4.0f, 6.0f};
        float[] result = controller.predict(input);
        assertArrayEquals(expected, result);
    }
}
