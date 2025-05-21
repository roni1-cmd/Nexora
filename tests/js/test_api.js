const request = require('supertest');
const app = require('../../src/js/api/server');

describe('API Tests', () => {
    it('should return predictions for valid input', async () => {
        const response = await request(app)
            .post('/api/predict')
            .send({ input: [1, 2, 3] });
        expect(response.status).toBe(200);
        expect(response.body.predictions).toEqual([2, 4, 6]);
    });
});
